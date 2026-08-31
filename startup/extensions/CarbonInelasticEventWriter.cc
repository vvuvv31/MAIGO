// Extra Class for CarbonInelasticEventWriter

#include "CarbonInelasticEventWriter.hh"

#include "CarbonInelasticTargetIdentity.hh"

#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicProcess.hh"
#include "G4Ions.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleChange.hh"
#include "G4ParticleDefinition.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#if __has_include("G4Version.hh")
#include "G4Version.hh"
#endif

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifndef CARBON_GIT_SHA
#define CARBON_GIT_SHA "unknown"
#endif
#ifndef CARBON_GIT_DIRTY
#define CARBON_GIT_DIRTY "unknown"
#endif
#ifndef CARBON_SOURCE_TREE
#define CARBON_SOURCE_TREE "unknown"
#endif

namespace {

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "CINEL02 writer requires a little-endian host");
#endif

constexpr std::int32_t direct_secondary = 0;
constexpr std::int32_t unsupported_but_recorded = 2;

[[noreturn]] void Fatal(const char* where, const std::string& message) {
    G4ExceptionDescription description;
    description << message;
    G4Exception(where, "CINEL02", FatalException, description);
    throw std::runtime_error(message);
}

template <typename Value>
void CopyName(char* destination, std::size_t capacity, const Value& source,
              const char* label) {
    const std::string value = source;
    if (value.empty()) {
        Fatal("CarbonInelasticEventWriter", std::string("Empty ") + label);
    }
    if (value.size() >= capacity) {
        Fatal("CarbonInelasticEventWriter", std::string(label) + " exceeds CINEL02 field width");
    }
    std::memset(destination, 0, capacity);
    std::memcpy(destination, value.data(), value.size());
}

float ExcitationFromMass(const G4DynamicParticle& particle,
                         const G4ParticleDefinition& definition) {
    // For ions, GetMass() may use Geant4's A*m_u transport mass while the
    // definition carries the nuclear mass table.  Their difference is a
    // ground-state mass convention offset (about 3 MeV for C-12), not a
    // physical excitation.  Geant4 exposes the actual nuclear excitation on
    // G4Ions; use it whenever the definition is an ion.
    if (const auto* ion = dynamic_cast<const G4Ions*>(&definition);
        ion != nullptr) {
        return static_cast<float>(std::max(0.0, ion->GetExcitationEnergy() / MeV));
    }
    const auto excitation = particle.GetMass() - definition.GetPDGMass();
    return static_cast<float>(std::max(0.0, excitation / MeV));
}

int AtomicMass(const G4ParticleDefinition& definition) {
    const auto mass = definition.GetAtomicMass();
    return mass > 0 ? mass : definition.GetBaryonNumber();
}

bool TransportIdentity(const CarbonCinel02Product& product) {
    if (product.pdg == 22 || product.pdg == 2112) {
        return true;
    }
    if (product.z <= 0 || product.a < product.z) {
        return false;
    }
    // Keep the capture writer's transport-identification domain synchronized
    // with carbon-isotope-mass-v2.  The target material remains explicitly
    // H-1/O-16; this list covers captured final-state nuclei, not material
    // composition.  The package compiler and loader perform the authoritative
    // mass/excitation check using the same versioned table.
    const auto known_ion =
        (product.z == 1 && product.a >= 1 && product.a <= 3) ||
        (product.z == 2 && (product.a == 3 || product.a == 4 ||
                            product.a == 6 || product.a == 8)) ||
        (product.z == 3 && product.a >= 6 && product.a <= 9) ||
        (product.z == 4 && (product.a == 4 || product.a == 6 ||
                            product.a == 7 || (product.a >= 9 && product.a <= 12))) ||
        (product.z == 5 && (product.a == 8 || (product.a >= 10 && product.a <= 14))) ||
        (product.z == 6 && ((product.a >= 9 && product.a <= 16) || product.a == 18)) ||
        (product.z == 7 && product.a >= 12 && product.a <= 19) ||
        (product.z == 8 && product.a >= 13 && product.a <= 21) ||
        (product.z == 9 && product.a >= 17 && product.a <= 23) ||
        (product.z == 10 && product.a >= 16 && product.a <= 24) ||
        (product.z == 11 && product.a >= 20 && product.a <= 25) ||
        (product.z == 12 && product.a >= 19 && product.a <= 26) ||
        (product.z == 13 && product.a >= 22 && product.a <= 27) ||
        (product.z == 14 && product.a == 27);
    if (!known_ion) {
        return false;
    }
    const auto expected_pdg =
        1'000'000'000 + product.z * 10'000 + product.a * 10;
    if (product.z == 1 && product.a == 1 && product.pdg == 2212) {
        return product.excitation <= 1.0e-4F;
    }
    // The first replay stage transports only ground-state ions.  A valid
    // Geant4 isomer is still serialized losslessly, but it must be marked as
    // an explicit unsupported product because the EM+Elastic secondary state
    // does not carry excitation energy.
    return product.pdg == expected_pdg && product.excitation <= 1.0e-4F;
}

std::uint32_t Crc32(const char* bytes, std::size_t size) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= static_cast<std::uint8_t>(bytes[index]);
        for (int bit = 0; bit < 8; ++bit) {
            const auto mask = static_cast<std::uint32_t>(-(crc & 1U));
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

G4ThreeVector ProjectileLocalDirection(const G4ThreeVector& global_direction,
                                       const G4ThreeVector& incident_direction) {
    const auto axis = incident_direction.unit();
    if (axis.mag2() <= 0.0) {
        Fatal("CarbonInelasticEventWriter", "CINEL02 input direction is not usable");
    }
    // Keep this basis bit-for-bit aligned with rotate_local_direction() in
    // src/detail/sycl_device_math.inc: local z is the projectile axis, the
    // reference is chosen from x/y, and local y = z cross local x.  Using a
    // different transverse convention would rotate every non-axial product
    // during CINEL02 replay while leaving the stored global direction looking
    // superficially valid.
    const auto reference = std::abs(axis.x()) < 0.9
                               ? G4ThreeVector{1.0, 0.0, 0.0}
                               : G4ThreeVector{0.0, 1.0, 0.0};
    const auto local_x = (reference - axis * reference.dot(axis)).unit();
    const auto local_y = axis.cross(local_x).unit();
    return G4ThreeVector{global_direction.dot(local_x),
                         global_direction.dot(local_y),
                         global_direction.dot(axis)};
}

std::string ThreadLabel() {
    const auto thread = G4Threading::G4GetThreadId();
    if (thread < 0) {
        return "master";
    }
    return std::to_string(thread);
}

const char* Environment(const char* name) {
    const auto* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : "unknown";
}

bool CanonicalUuid(const std::string& value) {
    if (value.size() != 36U || value[8] != '-' || value[13] != '-' ||
        value[18] != '-' || value[23] != '-') {
        return false;
    }
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (index == 8U || index == 13U || index == 18U || index == 23U) {
            continue;
        }
        if (!std::isxdigit(static_cast<unsigned char>(value[index]))) {
            return false;
        }
    }
    return true;
}

bool EnvironmentBool(const char* name) {
    const auto* value = std::getenv(name);
    if (value == nullptr || *value == '\0' || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "0") == 0 || std::strcmp(value, "no") == 0) {
        return false;
    }
    if (std::strcmp(value, "true") == 0 || std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "yes") == 0) {
        return true;
    }
    Fatal("CarbonInelasticEventWriter",
          std::string(name) + " must be true/false when set");
}

std::string JsonQuote(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const auto character : value) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(character) < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(
                              static_cast<unsigned char>(character))
                       << std::dec << std::setfill(' ');
            } else {
                output << character;
            }
            break;
        }
    }
    output << '"';
    return output.str();
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        Fatal("CarbonInelasticEventWriter", "Cannot read " + path.string());
    }
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

#if defined(__unix__) || defined(__APPLE__)
class ContractLock {
public:
    explicit ContractLock(const std::filesystem::path& contract) {
        const auto lock_path = contract.string() + ".lock";
        descriptor_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
        if (descriptor_ < 0 || ::flock(descriptor_, LOCK_EX) != 0) {
            if (descriptor_ >= 0) {
                ::close(descriptor_);
            }
            Fatal("CarbonInelasticEventWriter",
                  "Cannot lock contract " + contract.string() + ": " +
                      std::strerror(errno));
        }
    }

    ContractLock(const ContractLock&) = delete;
    ContractLock& operator=(const ContractLock&) = delete;

    ~ContractLock() {
        if (descriptor_ >= 0) {
            ::flock(descriptor_, LOCK_UN);
            ::close(descriptor_);
        }
    }

private:
    int descriptor_{-1};
};

void SyncDirectory(const std::filesystem::path& path) {
    const auto directory = path.parent_path().empty() ? std::filesystem::path{"."}
                                                        : path.parent_path();
    const auto descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        Fatal("CarbonInelasticEventWriter",
              "Cannot open contract directory for fsync " + directory.string() + ": " +
                  std::strerror(errno));
    }
    const auto sync_failed = ::fsync(descriptor) != 0;
    const auto close_failed = ::close(descriptor) != 0;
    if (sync_failed || close_failed) {
        Fatal("CarbonInelasticEventWriter",
              "Cannot fsync contract directory " + directory.string() + ": " +
                  std::strerror(errno));
    }
}

void SyncFile(const std::filesystem::path& path) {
    const auto descriptor = ::open(path.c_str(), O_WRONLY);
    if (descriptor < 0) {
        Fatal("CarbonInelasticEventWriter",
              "Cannot reopen raw file for fsync " + path.string() + ": " +
                  std::strerror(errno));
    }
    const auto sync_failed = ::fsync(descriptor) != 0;
    const auto close_failed = ::close(descriptor) != 0;
    if (sync_failed || close_failed) {
        Fatal("CarbonInelasticEventWriter",
              "Cannot fsync raw file " + path.string() + ": " + std::strerror(errno));
    }
}
#else
class ContractLock {
public:
    explicit ContractLock(const std::filesystem::path&) {}
};

void SyncDirectory(const std::filesystem::path&) {}
void SyncFile(const std::filesystem::path&) {}
#endif

void AtomicWriteText(const std::filesystem::path& path, const std::string& content,
                     const bool allow_overwrite) {
    std::filesystem::create_directories(path.parent_path());
    const ContractLock lock(path);
    if (std::filesystem::exists(path)) {
        if (ReadText(path) == content) {
            return;
        }
        if (!allow_overwrite) {
            Fatal("CarbonInelasticEventWriter",
                  "Existing contract disagrees with this campaign: " + path.string());
        }
    }

    static std::atomic<std::uint64_t> sequence{0};
    const auto temporary = std::filesystem::path(
        path.string() + ".tmp." + std::to_string(
                                  static_cast<unsigned long long>(sequence.fetch_add(1))) +
        "." + std::to_string(static_cast<unsigned long long>(
#if defined(__unix__) || defined(__APPLE__)
                                  ::getpid()
#else
                                  0
#endif
                                  )));
#if defined(__unix__) || defined(__APPLE__)
    const auto descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        Fatal("CarbonInelasticEventWriter",
              "Cannot create contract temporary file " + temporary.string());
    }
    std::size_t offset = 0;
    while (offset < content.size()) {
        const auto written = ::write(descriptor, content.data() + offset,
                                     content.size() - offset);
        if (written <= 0) {
            ::close(descriptor);
            std::filesystem::remove(temporary);
            Fatal("CarbonInelasticEventWriter",
                  "Cannot write contract temporary file " + temporary.string());
        }
        offset += static_cast<std::size_t>(written);
    }
    const auto sync_failed = ::fsync(descriptor) != 0;
    const auto close_failed = ::close(descriptor) != 0;
    if (sync_failed || close_failed) {
        std::filesystem::remove(temporary);
        Fatal("CarbonInelasticEventWriter",
              "Cannot fsync contract temporary file " + temporary.string());
    }
#else
    {
        std::ofstream output(temporary, std::ios::out | std::ios::trunc);
        if (!output) {
            Fatal("CarbonInelasticEventWriter",
                  "Cannot create contract temporary file " + temporary.string());
        }
        output << content;
        output.flush();
        if (!output) {
            std::filesystem::remove(temporary);
            Fatal("CarbonInelasticEventWriter",
                  "Cannot write contract temporary file " + temporary.string());
        }
    }
#endif
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        Fatal("CarbonInelasticEventWriter",
              "Cannot atomically publish contract " + path.string() + ": " +
                  error.message());
    }
    SyncDirectory(path);
}

std::string ExecutablePath() {
#if defined(__unix__) || defined(__APPLE__)
    std::array<char, 4096> buffer{};
    const auto length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1U);
    if (length > 0) {
        buffer[static_cast<std::size_t>(length)] = '\0';
        return buffer.data();
    }
#endif
    return "unknown";
}

std::string Sha256File(const std::filesystem::path& path) {
    constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    const auto rotate = [](const std::uint32_t value, const unsigned amount) {
        return (value >> amount) | (value << (32U - amount));
    };
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                        0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                        0x1f83d9abU, 0x5be0cd19U};
    const auto process = [&](const std::uint8_t* data) {
        std::array<std::uint32_t, 64> words{};
        for (int index = 0; index < 16; ++index) {
            words[index] = (static_cast<std::uint32_t>(data[4 * index]) << 24U) |
                           (static_cast<std::uint32_t>(data[4 * index + 1]) << 16U) |
                           (static_cast<std::uint32_t>(data[4 * index + 2]) << 8U) |
                           static_cast<std::uint32_t>(data[4 * index + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const auto s0 = rotate(words[index - 15], 7U) ^
                            rotate(words[index - 15], 18U) ^
                            (words[index - 15] >> 3U);
            const auto s1 = rotate(words[index - 2], 17U) ^
                            rotate(words[index - 2], 19U) ^
                            (words[index - 2] >> 10U);
            words[index] = s1 + words[index - 7] + s0 + words[index - 16];
        }
        auto a = state[0];
        auto b = state[1];
        auto c = state[2];
        auto d = state[3];
        auto e = state[4];
        auto f = state[5];
        auto g = state[6];
        auto h = state[7];
        for (int index = 0; index < 64; ++index) {
            const auto s1 = rotate(e, 6U) ^ rotate(e, 11U) ^ rotate(e, 25U);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto t1 = h + s1 + choose + constants[index] + words[index];
            const auto s0 = rotate(a, 2U) ^ rotate(a, 13U) ^ rotate(a, 22U);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    };
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Fatal("CarbonInelasticEventWriter", "Cannot hash " + path.string());
    }
    std::array<std::uint8_t, 64> buffer{};
    std::uint64_t bytes = 0;
    while (input.read(reinterpret_cast<char*>(buffer.data()), buffer.size())) {
        process(buffer.data());
        bytes += buffer.size();
    }
    const auto remainder = static_cast<std::size_t>(input.gcount());
    bytes += remainder;
    buffer[remainder] = 0x80U;
    if (remainder >= 56U) {
        std::fill(buffer.begin() + static_cast<std::ptrdiff_t>(remainder + 1U),
                  buffer.end(), 0U);
        process(buffer.data());
        buffer.fill(0U);
    } else {
        std::fill(buffer.begin() + static_cast<std::ptrdiff_t>(remainder + 1U),
                  buffer.end(), 0U);
    }
    const auto bits = bytes * 8U;
    for (int index = 0; index < 8; ++index) {
        buffer[56U + static_cast<std::size_t>(index)] =
            static_cast<std::uint8_t>(bits >> (56U - 8U * static_cast<unsigned>(index)));
    }
    process(buffer.data());
    std::ostringstream output;
    for (const auto word : state) {
        output << std::hex << std::setfill('0') << std::setw(8) << word;
    }
    return output.str();
}

}  // namespace

CarbonInelasticEventWriter& CarbonInelasticEventWriter::Instance() {
    static thread_local CarbonInelasticEventWriter writer;
    return writer;
}

CarbonInelasticEventWriter::CarbonInelasticEventWriter() {
    const char* configured = std::getenv("CARBON_CINEL02_OUTPUT_DIR");
    const char* configured_uuid = std::getenv("CARBON_CINEL02_CAMPAIGN_UUID");
    if (configured_uuid == nullptr || *configured_uuid == '\0' ||
        !CanonicalUuid(configured_uuid)) {
        Fatal("CarbonInelasticEventWriter",
              "CARBON_CINEL02_CAMPAIGN_UUID must be a canonical UUID");
    }
    campaign_uuid_ = configured_uuid;
    const auto base_directory = configured != nullptr && *configured != '\0'
                                    ? std::filesystem::path{configured}
                                    : std::filesystem::path{"cinel02_raw"};
    output_directory_ = (base_directory / campaign_uuid_).string();
    overwrite_campaign_ = EnvironmentBool("CARBON_CINEL02_OVERWRITE_CAMPAIGN");
}

CarbonInelasticEventWriter::~CarbonInelasticEventWriter() {
    try {
        Finalize();
    } catch (const std::exception& error) {
        // Contract/hash publication is part of the authoritative campaign
        // output. Do not silently turn a failed finalization into a usable
        // looking raw file.
        G4ExceptionDescription description;
        description << error.what();
        G4Exception("CarbonInelasticEventWriter", "CINEL02", FatalException,
                    description);
    } catch (...) {
        G4ExceptionDescription description;
        description << "Unknown CINEL02 finalization failure";
        G4Exception("CarbonInelasticEventWriter", "CINEL02", FatalException,
                    description);
    }
}

void CarbonInelasticEventWriter::OpenIfNeeded() {
    if (opened_) {
        return;
    }
    std::filesystem::create_directories(output_directory_);
    const char* tag = std::getenv("CARBON_CINEL02_RUN_TAG");
    const std::string run_tag = tag != nullptr && *tag != '\0' ? std::string("_") + tag : "";
    output_file_ = (std::filesystem::path(output_directory_) /
                    ("worker_" + ThreadLabel() + run_tag + ".cinel02"))
                       .string();
    if (std::filesystem::exists(output_file_) && !overwrite_campaign_) {
        Fatal("CarbonInelasticEventWriter",
              "CINEL02 raw file already exists; choose a new campaign UUID or set "
              "CARBON_CINEL02_OVERWRITE_CAMPAIGN=true: " + output_file_);
    }
    WriteCampaignContract();
    stream_.open(output_file_, std::ios::binary | std::ios::in | std::ios::out |
                              std::ios::trunc);
    if (!stream_) {
        Fatal("CarbonInelasticEventWriter", "Cannot open " + output_file_);
    }
    CarbonCinel02RawHeader header{};
    std::memcpy(header.magic, "CINEL02\0", sizeof(header.magic));
    stream_.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!stream_) {
        Fatal("CarbonInelasticEventWriter", "Cannot write CINEL02 header to " + output_file_);
    }
    opened_ = true;
}

void CarbonInelasticEventWriter::WriteCampaignContract() const {
    const auto path = std::filesystem::path(output_directory_) / "cinel02.contract.json";
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": {\"name\": \"CINEL02\", \"version\": 2},\n"
           << "  \"collision_state_source\": {\n"
           << "    \"authoritative\": true,\n"
           << "    \"object\": \"G4Track passed to wrapped PostStepDoIt\",\n"
           << "    \"pre_step_authoritative\": false\n"
           << "  },\n"
           << "  \"final_state_source\": {\n"
           << "    \"object\": \"G4VParticleChange returned by the same delegated PostStepDoIt\",\n"
           << "    \"delayed_secondary_reconstruction\": false\n"
           << "  },\n"
           << "  \"local_deposit\": {\n"
           << "    \"definition\": \"G4VParticleChange::GetLocalEnergyDeposit\",\n"
           << "    \"runtime_safe\": true\n"
           << "  },\n"
           << "  \"target_source\": \"G4HadronicProcess::GetTargetIsotope, with authoritative G4Nucleus Z/A fallback\",\n"
           << "  \"target_isotope_pointer_missing_policy\": {\n"
           << "    \"raw_target_isotope_id\": 4294967295,\n"
           << "    \"name_prefix\": \"G4Nucleus(Z=...\",\n"
           << "    \"z_a_are_authoritative\": true,\n"
           << "    \"counted_in_worker_contract\": true\n"
           << "  },\n"
           << "  \"depth_conditioned_production\": false,\n"
           << "  \"scalar_energy_scaling\": false,\n"
           << "  \"nearest_fill\": false,\n"
           << "  \"first_step_products\": false,\n"
           << "  \"whole_step_local_deposit\": false,\n"
           << "  \"capture_scope\": {\n"
           << "    \"primary_only\": "
           << JsonQuote(Environment("CARBON_CINEL02_PRIMARY_ONLY")) << "\n"
           << "  },\n"
           << "  \"campaign\": {\n"
           << "    \"uuid\": " << JsonQuote(campaign_uuid_) << ",\n"
           << "    \"output_directory\": " << JsonQuote(output_directory_) << ",\n"
           << "    \"raw_file_pattern\": \"worker_*.cinel02\",\n"
           << "    \"raw_contract_pattern\": \"worker_*.cinel02.contract.json\"\n"
           << "  },\n"
           << "  \"provenance\": {\n"
           << "    \"git_sha\": " << JsonQuote(CARBON_GIT_SHA) << ",\n"
           << "    \"git_dirty\": " << JsonQuote(CARBON_GIT_DIRTY) << ",\n"
           << "    \"executable\": " << JsonQuote(ExecutablePath()) << ",\n"
           << "    \"executable_sha256\": "
           << JsonQuote(Sha256File(ExecutablePath())) << ",\n"
           << "    \"source_tree\": " << JsonQuote(CARBON_SOURCE_TREE) << ",\n"
           << "    \"compiler\": " << JsonQuote(__VERSION__) << ",\n"
           << "    \"cxx_standard\": " << JsonQuote(std::to_string(__cplusplus))
           << ",\n"
           << "    \"topas_version\": " << JsonQuote(Environment("CARBON_TOPAS_VERSION"))
           << ",\n"
           << "    \"geant4_version\": " << JsonQuote(
#ifdef G4VERSION_TAG
                  G4VERSION_TAG
#elif defined(G4VERSION_NUMBER)
                  std::to_string(G4VERSION_NUMBER)
#else
                  Environment("CARBON_GEANT4_VERSION")
#endif
              ) << ",\n"
           << "    \"physics_list\": " << JsonQuote(Environment("CARBON_PHYSICS_LIST"))
           << ",\n"
           << "    \"production_cuts\": "
           << JsonQuote(Environment("CARBON_PRODUCTION_CUTS")) << ",\n"
           << "    \"step_limits\": " << JsonQuote(Environment("CARBON_STEP_LIMITS"))
           << ",\n"
           << "    \"random_seed_policy\": "
           << JsonQuote(Environment("CARBON_RANDOM_SEED_POLICY")) << ",\n"
           << "    \"capture_extension\": \"CarbonInelasticCaptureProcess/CINEL02\"\n"
           << "  }\n"
           << "}\n";
    AtomicWriteText(path, output.str(), overwrite_campaign_);
}

void CarbonInelasticEventWriter::WriteWorkerContract() const {
    const auto raw_path = std::filesystem::path(output_file_);
    const auto contract_path = std::filesystem::path(output_file_ + ".contract.json");
    const auto bytes = std::filesystem::file_size(raw_path);
    std::ostringstream output;
    output << "{\n"
           << "  \"schema\": \"CINEL02_WORKER_CONTRACT_V1\",\n"
           << "  \"campaign_uuid\": " << JsonQuote(campaign_uuid_) << ",\n"
           << "  \"worker\": " << JsonQuote(ThreadLabel()) << ",\n"
           << "  \"raw_file\": " << JsonQuote(raw_path.filename().string()) << ",\n"
           << "  \"raw_bytes\": " << bytes << ",\n"
           << "  \"target_isotope_pointer_missing_count\": "
           << target_isotope_pointer_missing_count_ << ",\n"
           << "  \"raw_sha256\": " << JsonQuote(Sha256File(raw_path)) << "\n"
           << "}\n";
    AtomicWriteText(contract_path, output.str(), overwrite_campaign_);
}

std::uint32_t CarbonInelasticEventWriter::NextInteractionSequence(
    std::uint64_t event_id, std::uint32_t track_id) {
    if (event_id != current_event_id_) {
        current_event_id_ = event_id;
        interaction_sequences_.clear();
    }
    auto& sequence = interaction_sequences_[track_id];
    return sequence++;
}

CarbonCinel02InputSnapshot CarbonInelasticEventWriter::CaptureInput(
    const G4Track& track, const G4Step& step) {
    const auto* definition = track.GetParticleDefinition();
    const auto* dynamic = track.GetDynamicParticle();
    const auto* pre = step.GetPreStepPoint();
    const auto* material = track.GetMaterial() != nullptr
                               ? track.GetMaterial()
                               : (pre != nullptr ? pre->GetMaterial() : nullptr);
    if (definition == nullptr || dynamic == nullptr || material == nullptr) {
        Fatal("CarbonInelasticEventWriter",
              "CINEL02 capture has incomplete track/material input state");
    }
    const auto* run_manager = G4RunManager::GetRunManager();
    const auto* run = run_manager != nullptr ? run_manager->GetCurrentRun() : nullptr;
    const auto* event = run_manager != nullptr ? run_manager->GetCurrentEvent() : nullptr;
    CarbonCinel02InputSnapshot input{};
    input.run_id = run != nullptr ? static_cast<std::uint64_t>(run->GetRunID()) : 0;
    input.thread_id = static_cast<std::uint32_t>(std::max(0, G4Threading::G4GetThreadId()));
    input.event_id = event != nullptr ? static_cast<std::uint64_t>(event->GetEventID()) : 0;
    input.track_id = static_cast<std::uint32_t>(std::max(0, track.GetTrackID()));
    input.parent_track_id = static_cast<std::uint32_t>(std::max(0, track.GetParentID()));
    input.projectile_pdg = definition->GetPDGEncoding();
    input.projectile_z = static_cast<std::int16_t>(definition->GetAtomicNumber());
    input.projectile_a = static_cast<std::int16_t>(AtomicMass(*definition));
    input.projectile_charge = static_cast<float>(definition->GetPDGCharge() / eplus);
    input.projectile_rest_mass = static_cast<float>(dynamic->GetMass() / MeV);
    input.projectile_excitation = ExcitationFromMass(*dynamic, *definition);
    input.collision_energy_MeV = static_cast<float>(track.GetKineticEnergy() / MeV);
    input.collision_energy_MeV_per_u = input.projectile_a > 0
                                           ? input.collision_energy_MeV / input.projectile_a
                                           : input.collision_energy_MeV;
    input.collision_position = track.GetPosition();
    input.collision_direction = track.GetMomentumDirection();
    input.collision_time_ns = static_cast<float>(track.GetGlobalTime() / ns);
    input.proper_time_ns = static_cast<float>(track.GetProperTime() / ns);
    input.track_weight = static_cast<float>(track.GetWeight());
    input.step_length_mm = static_cast<float>(step.GetStepLength() / mm);
    input.material_id = static_cast<std::int32_t>(material->GetIndex());
    const auto* couple = track.GetMaterialCutsCouple() != nullptr
                             ? track.GetMaterialCutsCouple()
                             : (pre != nullptr ? pre->GetMaterialCutsCouple() : nullptr);
    input.cuts_couple_id = couple != nullptr ? couple->GetIndex() : -1;
    input.source_initial_energy_MeV =
        static_cast<float>(track.GetVertexKineticEnergy() / MeV);
    input.absolute_depth_mm = static_cast<float>(input.collision_position.z() / mm);
    input.material_name = material->GetName();
    if (pre != nullptr) {
        input.audit_pre_energy_MeV = static_cast<float>(pre->GetKineticEnergy() / MeV);
        input.audit_pre_position = pre->GetPosition();
        input.audit_pre_direction = pre->GetMomentumDirection();
    }
    input.audit_whole_step_deposit_MeV =
        static_cast<float>(step.GetTotalEnergyDeposit() / MeV);
    return input;
}

void CarbonInelasticEventWriter::Append(const CarbonCinel02InputSnapshot& input,
                                        const G4ParticleChange& change,
                                        const G4VProcess& process,
                                        G4HadronicProcess* hadronic) {
    if (hadronic == nullptr) {
        Fatal("CarbonInelasticEventWriter", "CINEL02 capture received a non-hadronic process");
    }
    const auto target = ResolveCarbonCinel02Target(hadronic);
    if (target.z <= 0 || target.a < target.z || target.name.empty()) {
        Fatal("CarbonInelasticEventWriter",
              "ion-inelastic PostStepDoIt returned no usable target nucleus identity");
    }
    if (!target.isotope_pointer_present) {
        ++target_isotope_pointer_missing_count_;
    }
    const auto* model = hadronic->GetHadronicInteraction();
    if (model == nullptr || model->GetModelName().empty()) {
        Fatal("CarbonInelasticEventWriter",
              "ion-inelastic PostStepDoIt returned no hadronic model identity");
    }
    CarbonCinel02Interaction record{};
    record.run_id = input.run_id;
    record.thread_id = input.thread_id;
    record.event_id = input.event_id;
    record.track_id = input.track_id;
    record.parent_track_id = input.parent_track_id;
    record.interaction_sequence = NextInteractionSequence(input.event_id, input.track_id);
    record.projectile_pdg = input.projectile_pdg;
    record.projectile_z = input.projectile_z;
    record.projectile_a = input.projectile_a;
    record.projectile_charge = input.projectile_charge;
    record.projectile_rest_mass = input.projectile_rest_mass;
    record.projectile_excitation = input.projectile_excitation;
    record.collision_energy_MeV = input.collision_energy_MeV;
    record.collision_energy_MeV_per_u = input.collision_energy_MeV_per_u;
    record.collision_x_mm = static_cast<float>(input.collision_position.x() / mm);
    record.collision_y_mm = static_cast<float>(input.collision_position.y() / mm);
    record.collision_z_mm = static_cast<float>(input.collision_position.z() / mm);
    record.collision_direction_x = static_cast<float>(input.collision_direction.x());
    record.collision_direction_y = static_cast<float>(input.collision_direction.y());
    record.collision_direction_z = static_cast<float>(input.collision_direction.z());
    record.collision_time_ns = input.collision_time_ns;
    record.proper_time_ns = input.proper_time_ns;
    record.track_weight = input.track_weight;
    record.step_length_mm = input.step_length_mm;
    record.material_id = input.material_id;
    record.cuts_couple_id = input.cuts_couple_id;
    record.target_z = static_cast<std::int16_t>(target.z);
    record.target_a = static_cast<std::int16_t>(target.a);
    record.target_isotope_id = target.isotope_pointer_present
                                   ? static_cast<std::uint32_t>(target.isotope_id)
                                   : std::numeric_limits<std::uint32_t>::max();
    record.process_type = process.GetProcessType();
    record.process_subtype = process.GetProcessSubType();
    record.model_id = -1;
    record.parent_status = static_cast<std::int32_t>(change.GetTrackStatus());
    record.parent_pdg = input.projectile_pdg;
    record.parent_z = record.projectile_z;
    record.parent_a = record.projectile_a;
    record.parent_charge = input.projectile_charge;
    record.parent_rest_mass = static_cast<float>(change.GetMass() / MeV);
    record.parent_excitation = record.projectile_excitation;
    record.parent_energy_MeV = static_cast<float>(change.GetEnergy() / MeV);
    const auto* parent_direction = change.GetMomentumDirection();
    const auto* parent_position = change.GetPosition();
    if (parent_direction == nullptr || parent_position == nullptr) {
        Fatal("CarbonInelasticEventWriter", "ParticleChange did not expose parent final state");
    }
    record.parent_direction_x = static_cast<float>(parent_direction->x());
    record.parent_direction_y = static_cast<float>(parent_direction->y());
    record.parent_direction_z = static_cast<float>(parent_direction->z());
    record.parent_x_mm = static_cast<float>(parent_position->x() / mm);
    record.parent_y_mm = static_cast<float>(parent_position->y() / mm);
    record.parent_z_mm = static_cast<float>(parent_position->z() / mm);
    record.parent_time_ns = static_cast<float>(change.GetGlobalTime() / ns);
    record.parent_proper_time_ns = static_cast<float>(change.GetProperTime() / ns);
    record.parent_weight = static_cast<float>(change.GetParentWeight());
    record.process_local_deposit_MeV = static_cast<float>(change.GetLocalEnergyDeposit() / MeV);
    record.nonionizing_deposit_MeV = static_cast<float>(change.GetNonIonizingEnergyDeposit() / MeV);
    record.source_initial_energy_MeV = input.source_initial_energy_MeV;
    record.absolute_depth_mm = input.absolute_depth_mm;
    CopyName(record.material_name, sizeof(record.material_name), input.material_name,
             "material name");
    CopyName(record.target_isotope_name, sizeof(record.target_isotope_name), target.name, "target isotope name");
    CopyName(record.process_name, sizeof(record.process_name), process.GetProcessName(), "process name");
    CopyName(record.model_name, sizeof(record.model_name), model->GetModelName(), "model name");

    record.audit_pre_energy_MeV = input.audit_pre_energy_MeV;
    record.audit_pre_direction_x = static_cast<float>(input.audit_pre_direction.x());
    record.audit_pre_direction_y = static_cast<float>(input.audit_pre_direction.y());
    record.audit_pre_direction_z = static_cast<float>(input.audit_pre_direction.z());
    record.audit_pre_x_mm = static_cast<float>(input.audit_pre_position.x() / mm);
    record.audit_pre_y_mm = static_cast<float>(input.audit_pre_position.y() / mm);
    record.audit_pre_z_mm = static_cast<float>(input.audit_pre_position.z() / mm);
    record.audit_whole_step_deposit_MeV = input.audit_whole_step_deposit_MeV;

    if (!std::isfinite(record.collision_energy_MeV_per_u) || record.collision_energy_MeV_per_u <= 0.0F ||
        !std::isfinite(record.process_local_deposit_MeV) || record.process_local_deposit_MeV < 0.0F ||
        !std::isfinite(record.nonionizing_deposit_MeV) || record.nonionizing_deposit_MeV < 0.0F ||
        !std::isfinite(record.parent_energy_MeV) || record.parent_energy_MeV < 0.0F ||
        !std::isfinite(record.track_weight) || std::abs(record.track_weight - 1.0F) > 1.0e-6F ||
        !std::isfinite(record.parent_weight) || std::abs(record.parent_weight - 1.0F) > 1.0e-6F ||
        (record.parent_status != fAlive && record.parent_status != fStopAndKill)) {
        Fatal("CarbonInelasticEventWriter", "CINEL02 captured non-finite process state");
    }

    if (record.parent_status == fStopAndKill && record.parent_energy_MeV > 1.0e-4F) {
        Fatal("CarbonInelasticEventWriter",
              "CINEL02 stop-and-kill parent retained non-zero kinetic energy");
    }

    // G4ParticleChange does not expose a replacement PDG definition.  A
    // surviving parent is therefore only safe to replay when the observable
    // charge and mass remain those of the authoritative input projectile;
    // otherwise the package would silently relabel a changed parent as C-12.
    if (record.parent_status == fAlive) {
        const auto output_charge = static_cast<float>(change.GetCharge() / eplus);
        const auto output_mass = static_cast<float>(change.GetMass() / MeV);
        const auto charge_tolerance = 1.0e-3F * std::max(1.0F, std::abs(input.projectile_charge));
        const auto mass_tolerance = 1.0e-3F * std::max(1.0F, std::abs(input.projectile_rest_mass));
        if (!std::isfinite(output_charge) || !std::isfinite(output_mass) ||
            std::abs(output_charge - input.projectile_charge) > charge_tolerance ||
            std::abs(output_mass - input.projectile_rest_mass) > mass_tolerance) {
            Fatal("CarbonInelasticEventWriter",
                  "CINEL02 surviving parent changed projectile charge or mass identity");
        }
    }

    std::vector<CarbonCinel02Product> products;
    products.reserve(static_cast<std::size_t>(change.GetNumberOfSecondaries()));
    for (G4int index = 0; index < change.GetNumberOfSecondaries(); ++index) {
        const auto* secondary = change.GetSecondary(index);
        if (secondary == nullptr || secondary->GetDynamicParticle() == nullptr ||
            secondary->GetDefinition() == nullptr) {
            Fatal("CarbonInelasticEventWriter", "ParticleChange contains a null direct secondary");
        }
        const auto* secondary_definition = secondary->GetDefinition();
        const auto* secondary_dynamic = secondary->GetDynamicParticle();
        CarbonCinel02Product product{};
        product.pdg = secondary_definition->GetPDGEncoding();
        product.z = static_cast<std::int16_t>(secondary_definition->GetAtomicNumber());
        product.a = static_cast<std::int16_t>(AtomicMass(*secondary_definition));
        product.charge = static_cast<float>(secondary_definition->GetPDGCharge() / eplus);
        product.rest_mass = static_cast<float>(secondary_dynamic->GetMass() / MeV);
        product.excitation = ExcitationFromMass(*secondary_dynamic, *secondary_definition);
        product.kinetic_energy_MeV = static_cast<float>(secondary->GetKineticEnergy() / MeV);
        const auto& direction = secondary->GetMomentumDirection();
        const auto local_direction = ProjectileLocalDirection(direction, input.collision_direction);
        const auto& position = secondary->GetPosition();
        product.direction_x = static_cast<float>(direction.x());
        product.direction_y = static_cast<float>(direction.y());
        product.direction_z = static_cast<float>(direction.z());
        product.local_direction_x = static_cast<float>(local_direction.x());
        product.local_direction_y = static_cast<float>(local_direction.y());
        product.local_direction_z = static_cast<float>(local_direction.z());
        product.position_x_mm = static_cast<float>(position.x() / mm);
        product.position_y_mm = static_cast<float>(position.y() / mm);
        product.position_z_mm = static_cast<float>(position.z() / mm);
        product.creation_time_ns = static_cast<float>(secondary->GetGlobalTime() / ns);
        product.weight = static_cast<float>(secondary->GetWeight());
        if (!std::isfinite(product.kinetic_energy_MeV) || product.kinetic_energy_MeV < 0.0F ||
            !std::isfinite(product.weight) || std::abs(product.weight - 1.0F) > 1.0e-6F ||
            !std::isfinite(product.direction_x) || !std::isfinite(product.direction_y) ||
            !std::isfinite(product.direction_z) || !std::isfinite(product.local_direction_x) ||
            !std::isfinite(product.local_direction_y) || !std::isfinite(product.local_direction_z)) {
            Fatal("CarbonInelasticEventWriter", "CINEL02 captured non-finite product state");
        }
        product.role = TransportIdentity(product) ? direct_secondary : unsupported_but_recorded;
        if (product.role == unsupported_but_recorded) {
            ++record.unsupported_product_count;
            record.unsupported_product_energy_MeV += product.kinetic_energy_MeV;
        }
        products.push_back(product);
    }
    if (products.size() > std::numeric_limits<std::uint32_t>::max()) {
        Fatal("CarbonInelasticEventWriter", "CINEL02 direct-product count exceeds uint32");
    }
    record.direct_product_count = static_cast<std::uint32_t>(products.size());
    OpenIfNeeded();
    WriteRecord(record, products);
}

void CarbonInelasticEventWriter::WriteRecord(
    const CarbonCinel02Interaction& record,
    const std::vector<CarbonCinel02Product>& products) {
    const auto payload_size = sizeof(record) + products.size() * sizeof(CarbonCinel02Product);
    const auto record_size = sizeof(CarbonCinel02RecordPrefix) + payload_size + sizeof(std::uint32_t);
    if (payload_size > std::numeric_limits<std::uint32_t>::max() ||
        record_size > std::numeric_limits<std::uint32_t>::max()) {
        Fatal("CarbonInelasticEventWriter", "CINEL02 event record exceeds uint32 length");
    }
    CarbonCinel02RecordPrefix prefix{};
    std::memcpy(prefix.magic, "CIR2", sizeof(prefix.magic));
    prefix.record_length = static_cast<std::uint32_t>(record_size);
    prefix.payload_length = static_cast<std::uint32_t>(payload_size);
    stream_.write(reinterpret_cast<const char*>(&prefix), sizeof(prefix));
    stream_.write(reinterpret_cast<const char*>(&record), sizeof(record));
    for (const auto& product : products) {
        stream_.write(reinterpret_cast<const char*>(&product), sizeof(product));
    }
    std::vector<char> payload(payload_size);
    std::memcpy(payload.data(), &record, sizeof(record));
    if (!products.empty()) {
        std::memcpy(payload.data() + sizeof(record), products.data(),
                    products.size() * sizeof(CarbonCinel02Product));
    }
    const auto crc = Crc32(payload.data(), payload.size());
    stream_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    if (!stream_) {
        Fatal("CarbonInelasticEventWriter", "Failed writing CINEL02 record to " + output_file_);
    }
    ++interaction_count_;
    product_count_ += products.size();
    bytes_written_ += record_size;
}

void CarbonInelasticEventWriter::Finalize() {
    if (!opened_) {
        return;
    }
    stream_.flush();
    if (!stream_) {
        Fatal("CarbonInelasticEventWriter",
              "Failed flushing CINEL02 raw records to " + output_file_);
    }
    CarbonCinel02RawHeader header{};
    std::memcpy(header.magic, "CINEL02\0", sizeof(header.magic));
    header.interaction_count = interaction_count_;
    header.product_count = product_count_;
    header.bytes_written = bytes_written_;
    stream_.seekp(0);
    stream_.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream_.flush();
    if (!stream_) {
        Fatal("CarbonInelasticEventWriter",
              "Failed finalizing CINEL02 raw header in " + output_file_);
    }
    stream_.close();
    if (stream_.fail()) {
        Fatal("CarbonInelasticEventWriter",
              "Failed closing CINEL02 raw file " + output_file_);
    }
    opened_ = false;
    // std::fstream does not expose a portable native descriptor. Reopen the
    // fully written file solely to make the durability boundary explicit:
    // data, header, file metadata, then worker contract.
    SyncFile(output_file_);
    SyncDirectory(output_file_);
    WriteWorkerContract();
}
