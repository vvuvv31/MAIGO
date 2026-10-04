#pragma once
#include "carbon/material_electron_response.hpp"
#include "carbon/sha256.hpp"
#include <bit>
#include <type_traits>
#include <chrono>
#include <zstd.h>

// A prepared bank is a derived, immutable artifact, not a waiver of source
// validation. Only the offline compiler writes it, after the original loaders
// and genealogy/energy checks pass. Every block is hashed while being read.
// The algorithm fingerprint is generated from the compiler/loader sources.
#ifndef CARBON_MATERIAL_PREPARED_VERSION
#error "Missing material prepared-cache algorithm fingerprint"
#endif
namespace carbon {
inline std::string material_prepared_key(const MaterialElectronResponseIndex& index,std::size_t id,bool packed=false,bool compact=false) {
    const auto& f=index.files.at(id);
    std::ostringstream s;
    s<<(compact?"material-prepared-carbon-em-v7 ":packed?"material-prepared-proton-em-v2 ":"material-prepared-v1 ")<<CARBON_MATERIAL_PREPARED_VERSION<<' '
     <<index.projectile_z<<' '<<index.projectile_a<<' '
     <<f.data_sha256<<' '<<f.metadata_sha256<<' '
     <<f.state_reference_sha256<<' '<<f.state_reference_metadata_sha256<<' '
     <<sizeof(ElectronContinuationStep)<<' '<<alignof(ElectronContinuationStep)<<' '
     <<sizeof(ElectronBirthSample)<<' '<<sizeof(ElectronBirthChannel)<<' '
     <<sizeof(WaterElectronChannel)<<' '<<sizeof(std::size_t)<<' '
     <<(std::endian::native==std::endian::little);
    return compute_sha256_hex(s.str());
}

class MaterialPreparedDigest {
public:
#ifdef CARBON_OPENSSL_SHA256
    MaterialPreparedDigest():ctx_(EVP_MD_CTX_new(),EVP_MD_CTX_free) {
        if(!ctx_ || EVP_DigestInit_ex(ctx_.get(),EVP_sha256(),nullptr)!=1)
            throw std::runtime_error("Prepared SHA initialization failed");
    }
    void update(const void* p,std::size_t n) {
        if(EVP_DigestUpdate(ctx_.get(),p,n)!=1)throw std::runtime_error("Prepared SHA update failed");
    }
    std::string finish() {
        unsigned char digest[32];unsigned n=0;
        if(EVP_DigestFinal_ex(ctx_.get(),digest,&n)!=1 || n!=32)throw std::runtime_error("Prepared SHA finalize failed");
        std::ostringstream s;s<<std::hex<<std::setfill('0');
        for(auto b:digest)s<<std::setw(2)<<unsigned(b);
        return s.str();
    }
private:
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx_;
#else
    void update(const void* p,std::size_t n) {ctx_.update(p,n);}
    std::string finish() {return ctx_.finalize_hex();}
private:
    Sha256 ctx_;
#endif
};

class MaterialPreparedReader {
public:
    MaterialPreparedReader(const std::filesystem::path& root,const std::string& key,bool packed=false,bool compact=false):dir_(root/key) {
        std::ifstream pin(dir_/"manifest.sha256");std::string sha,extra;pin>>sha;
        if(!pin || sha.size()!=64 || (pin>>extra) || !file_sha256_matches(dir_/"manifest",sha))
            throw std::runtime_error("Missing or changed prepared manifest: "+dir_.string());
        in_.open(dir_/"manifest");std::string magic,identity;
        in_>>magic>>identity>>section>>density>>reference_density>>nodes>>sources;
        if(!in_ || magic!=(compact?"MAIGO_CARBON_EM_7":packed?"MAIGO_PROTON_EM_2":"MAIGO_PREPARED_1") || identity!=key ||
           !std::isfinite(density) || density<=0 || !std::isfinite(reference_density) || reference_density<=0 ||
           !nodes || !sources || sources>1024)
            throw std::runtime_error("Invalid prepared manifest identity/layout");
    }
    // Allocation goes directly to the final USM destination: no second giant
    // host vector. Allocator owns cleanup if reading or SHA validation fails.
    template<class T,class Allocate>
    const T* read(const std::string& expected,Allocate allocate,std::size_t& count) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::string name,sha;std::uint64_t n=0,width=0;
        in_>>name>>n>>width>>sha;
        if(!in_ || name!=expected || width!=sizeof(T) || sha.size()!=64 ||
           n>std::numeric_limits<std::size_t>::max()/sizeof(T))
            throw std::runtime_error("Invalid prepared block descriptor: "+expected);
        const auto path=dir_/(std::to_string(block_++)+".bin");
        const std::filesystem::path compressed_path=path.string()+".zst";
        const bool compressed=std::filesystem::exists(compressed_path);
        const auto bytes=static_cast<std::size_t>(n)*sizeof(T);
        if(compressed && std::filesystem::exists(path))throw std::runtime_error("Ambiguous prepared block encoding");
        if(!compressed && std::filesystem::file_size(path)!=bytes)throw std::runtime_error("Prepared block size mismatch");
        T* dest=allocate(static_cast<std::size_t>(n));MaterialPreparedDigest digest;
        if(compressed) {
            std::ifstream input(compressed_path,std::ios::binary);
            if(!input)throw std::runtime_error("Missing compressed prepared block");
            std::unique_ptr<ZSTD_DStream,decltype(&ZSTD_freeDStream)> decoder(ZSTD_createDStream(),ZSTD_freeDStream);
            if(!decoder || ZSTD_isError(ZSTD_initDStream(decoder.get())))throw std::runtime_error("Prepared decompressor initialization failed");
            // The response budget applies to the final uncompressed allocation.
            // Bound decoder auxiliary window memory even for a damaged frame.
            if(ZSTD_isError(ZSTD_DCtx_setParameter(decoder.get(),ZSTD_d_windowLogMax,27)))
                throw std::runtime_error("Prepared decompressor window limit failed");
            std::vector<char> encoded(8*1024*1024);std::size_t offset=0;bool finished=false;
            while(!finished && input) {
                input.read(encoded.data(),encoded.size());const auto got=input.gcount();
                if(input.bad())throw std::runtime_error("Compressed prepared read failed");
                if(got<=0)break;
                ZSTD_inBuffer in{encoded.data(),static_cast<std::size_t>(got),0};
                bool pending_output=false;
                while(in.pos<in.size || pending_output) {
                    unsigned char overflow{};
                    const auto available=bytes-offset;
                    void* target=available?static_cast<void*>(reinterpret_cast<char*>(dest)+offset):static_cast<void*>(&overflow);
                    ZSTD_outBuffer out{target,available?std::min<std::size_t>(8*1024*1024,available):std::size_t{1},0};
                    const auto previous=in.pos;
                    const auto remaining=ZSTD_decompressStream(decoder.get(),&out,&in);
                    if(ZSTD_isError(remaining) || out.pos>available)throw std::runtime_error("Invalid or oversized compressed prepared block");
                    digest.update(target,out.pos);offset+=out.pos;
                    pending_output=remaining!=0 && out.pos==out.size;
                    if(remaining==0) {
                        if(in.pos!=in.size || input.peek()!=std::char_traits<char>::eof())
                            throw std::runtime_error("Trailing compressed prepared data");
                        finished=true;break;
                    }
                    if(in.pos==previous && out.pos==0) {
                        // Draining a full output buffer may discover that the
                        // decoder needs the next input chunk. EOF is checked
                        // below; this is not an error with exhausted input.
                        if(in.pos==in.size) {pending_output=false;break;}
                        throw std::runtime_error("Compressed prepared block made no progress");
                    }
                }
            }
            if(!finished || offset!=bytes)throw std::runtime_error("Truncated compressed prepared block");
        } else {
            std::ifstream input(path,std::ios::binary);
            for(std::size_t offset=0;offset<bytes;) {
                const auto length=std::min<std::size_t>(8*1024*1024,bytes-offset);
                auto* p=reinterpret_cast<char*>(dest)+offset;
                if(!input.read(p,length))throw std::runtime_error("Truncated prepared block");
                digest.update(p,length);offset+=length;
            }
            if(!input)throw std::runtime_error("Prepared block read failed");
        }
        if(digest.finish()!=sha)throw std::runtime_error("Prepared block SHA mismatch");
        count=n;return dest;
    }
    void finish() {std::string extra;if(in_>>extra)throw std::runtime_error("Trailing prepared descriptors");}
    int section{};double density{},reference_density{};std::size_t nodes{},sources{};
private:
    std::filesystem::path dir_;std::ifstream in_;std::size_t block_{};
};

class MaterialPreparedWriter {
public:
    MaterialPreparedWriter(const std::filesystem::path& staging,const std::string& key,
        const WaterElectronResponseTable& table,double density,std::size_t sources,bool packed=false,bool compact=false):dir_(staging) {
        if(!std::filesystem::create_directory(dir_))throw std::runtime_error("Prepared staging directory already exists");
        out_.open(dir_/"manifest");
        out_<<(compact?"MAIGO_CARBON_EM_7 ":packed?"MAIGO_PROTON_EM_2 ":"MAIGO_PREPARED_1 ")<<key<<' '<<table.material_section<<' '<<std::setprecision(17)
            <<density<<' '<<table.reference_density_g_cm3<<' '<<table.nodes.size()<<' '<<sources<<'\n';
    }
    template<class T> void write(const std::string& name,const std::vector<T>& data) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::ofstream output(dir_/(std::to_string(block_++)+".bin"),std::ios::binary);
        MaterialPreparedDigest digest;const auto bytes=data.size()*sizeof(T);
        for(std::size_t offset=0;offset<bytes;) {
            const auto length=std::min<std::size_t>(8*1024*1024,bytes-offset);
            const auto* p=reinterpret_cast<const char*>(data.data())+offset;
            output.write(p,length);digest.update(p,length);offset+=length;
        }
        output.close();if(!output)throw std::runtime_error("Prepared block write failed");
        out_<<name<<' '<<data.size()<<' '<<sizeof(T)<<' '<<digest.finish()<<'\n';
        if(!out_)throw std::runtime_error("Prepared manifest write failed");
    }
    void finish() {
        out_.close();if(!out_)throw std::runtime_error("Prepared manifest close failed");
        std::ofstream pin(dir_/"manifest.sha256");pin<<compute_file_sha256_hex(dir_/"manifest")<<'\n';
        pin.close();if(!pin)throw std::runtime_error("Prepared pin write failed");
    }
private:
    std::filesystem::path dir_;std::ofstream out_;std::size_t block_{};
};
} // namespace carbon
