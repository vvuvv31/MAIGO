#!/usr/bin/env python3
"""Create a centered square TOPAS minibeam field from a larger spot grid."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


VECTOR_RE = re.compile(
    r"^(?P<prefix>[diu]v:Tf/Scatterer1/L(?P<index>\d+)/"
    r"(?P<kind>Times|Values)\s*=\s*)(?P<body>.*)$"
)


def split_vector(line: str) -> tuple[re.Match[str], list[str], str] | None:
    match = VECTOR_RE.match(line)
    if not match:
        return None
    tokens = match.group("body").split()
    try:
        count = int(tokens[0])
    except ValueError:
        # Keep alias declarations such as "L1/Times = L0/Times ms" unchanged.
        return None
    values = tokens[1 : count + 1]
    suffix = " ".join(tokens[count + 1 :])
    if len(values) != count:
        raise ValueError(f"Expected {count} values in {line[:80]!r}")
    return match, values, suffix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spots-input", type=Path, required=True)
    parser.add_argument("--run-input", type=Path, required=True)
    parser.add_argument("--spots-output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=15)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument(
        "--histories-per-spot",
        type=int,
        default=None,
        help="Override L4 histories after spatial subsetting",
    )
    parser.add_argument("--dose-output", default="topas/dose")
    parser.add_argument(
        "--static-sources-output",
        type=Path,
        default=None,
        help="Also emit all spots as static TOPAS sources in one run",
    )
    parser.add_argument(
        "--include-prefix",
        default="center15x15",
        help="Path from the TOPAS working directory to generated case files",
    )
    args = parser.parse_args()

    if args.side <= 0 or args.side % 2 == 0:
        raise ValueError("--side must be a positive odd integer")

    input_lines = args.spots_input.read_text(encoding="utf-8").splitlines()
    vectors: dict[tuple[int, str], list[str]] = {}
    for line in input_lines:
        parsed = split_vector(line)
        if parsed:
            match, values, _ = parsed
            vectors[(int(match.group("index")), match.group("kind"))] = values
    x = np.asarray(vectors[(5, "Values")], dtype=np.float64)
    y = np.asarray(vectors[(6, "Values")], dtype=np.float64)
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    if unique_x.size != unique_y.size or x.size != unique_x.size * unique_y.size:
        raise ValueError("Input is not a complete square spot grid")
    if args.side > unique_x.size:
        raise ValueError("--side exceeds the input grid")
    start = (unique_x.size - args.side) // 2
    selected_x = unique_x[start : start + args.side]
    selected_y = unique_y[start : start + args.side]
    selected = np.isin(x, selected_x) & np.isin(y, selected_y)
    indices = np.flatnonzero(selected)
    count = int(indices.size)
    if count != args.side * args.side:
        raise ValueError(f"Selected {count} spots, expected {args.side ** 2}")

    output_lines: list[str] = []
    for line in input_lines:
        parsed = split_vector(line)
        if not parsed:
            output_lines.append(line)
            continue
        match, values, suffix = parsed
        index = int(match.group("index"))
        kind = match.group("kind")
        if index == 0 and kind in ("Times", "Values"):
            chosen = [str(value) for value in range(1, count + 1)]
        elif index == 4 and kind == "Values" and args.histories_per_spot:
            chosen = [str(args.histories_per_spot)] * count
        else:
            chosen = [values[value] for value in indices]
        rendered = f"{match.group('prefix')}{count}\t" + "\t".join(chosen)
        if suffix:
            rendered += f"\t{suffix}"
        output_lines.append(rendered)

    run = args.run_input.read_text(encoding="utf-8")
    run = run.replace("includeFile = spots961.txt", (
        f"includeFile = {args.include_prefix}/{args.spots_output.name}"
    ))
    run = run.replace("includeFile = aperture.txt", (
        f"includeFile = {args.include_prefix}/aperture.txt"
    ))
    run = re.sub(
        r"^d:Tf/TimelineEnd\s*=\s*\d+\s*ms.*$",
        f"d:Tf/TimelineEnd = {count} ms",
        run,
        flags=re.MULTILINE,
    )
    run = re.sub(
        r"^i:Tf/NumberOfSequentialTimes\s*=\s*\d+.*$",
        f"i:Tf/NumberOfSequentialTimes = {count}",
        run,
        flags=re.MULTILINE,
    )
    run = re.sub(
        r"^i:Ts/NumberOfThreads\s*=\s*\d+.*$",
        f"i:Ts/NumberOfThreads = {args.threads}",
        run,
        flags=re.MULTILINE,
    )
    # This validation is physical-dose only. Remove the optional HadronLET scorer.
    run = re.sub(
        r"^s:Sc/LET/Quantity.*?^i:Sc/LET/ZBins\s*=\s*\d+\s*$\n?",
        "",
        run,
        flags=re.MULTILINE | re.DOTALL,
    )
    run = run.replace(
        's:Sc/DoseAtPhantomP/OutputFile = "dose"',
        f's:Sc/DoseAtPhantomP/OutputFile = '
        f'"{args.include_prefix}/{args.dose_output}"',
    )

    args.spots_output.parent.mkdir(parents=True, exist_ok=True)
    args.run_output.parent.mkdir(parents=True, exist_ok=True)
    args.spots_output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    if args.static_sources_output:
        subset: dict[int, list[str]] = {}
        for index in range(1, 15):
            source_values = vectors[(index, "Values")]
            subset[index] = [source_values[value] for value in indices]
        if args.histories_per_spot:
            subset[4] = [str(args.histories_per_spot)] * count
        source_lines = [
            "# Static-source expansion of the centered spot field.",
            "# All sources execute in one TOPAS run; physics/emittance is unchanged.",
            "",
        ]
        for spot in range(count):
            component = f"BeamPosition{spot + 1:03d}"
            source = f"CarbonSpot{spot + 1:03d}"
            source_lines.extend(
                (
                    f's:Ge/{component}/Parent = "World"',
                    f's:Ge/{component}/Type = "Group"',
                    f"d:Ge/{component}/TransX = {subset[5][spot]} mm",
                    f"d:Ge/{component}/TransY = -0.45 m",
                    f"d:Ge/{component}/TransZ = {subset[6][spot]} mm",
                    f"d:Ge/{component}/RotX = {subset[7][spot]} deg",
                    f"d:Ge/{component}/RotY = {subset[8][spot]} deg",
                    f"d:Ge/{component}/RotZ = 0 deg",
                    "",
                    f's:So/{source}/Type = "emittance"',
                    f's:So/{source}/Component = "{component}"',
                    f's:So/{source}/EmittanceParticle = "GenericIon(6,12,6)"',
                    f's:So/{source}/Distribution = "BiGaussian"',
                    f"d:So/{source}/EmittanceEnergy = {subset[2][spot]} MeV",
                    f"u:So/{source}/EmittanceEnergySpread = {subset[3][spot]}",
                    f"d:So/{source}/SigmaX = {subset[9][spot]} mm",
                    f"u:So/{source}/SigmaXprime = {subset[10][spot]}",
                    f"u:So/{source}/CorrelationX = {subset[11][spot]}",
                    f"d:So/{source}/SigmaY = {subset[12][spot]} mm",
                    f"u:So/{source}/SigmaYprime = {subset[13][spot]}",
                    f"u:So/{source}/CorrelationY = {subset[14][spot]}",
                    f"i:So/{source}/NumberOfHistoriesInRun = {subset[4][spot]}",
                    "",
                )
            )
        args.static_sources_output.parent.mkdir(parents=True, exist_ok=True)
        args.static_sources_output.write_text(
            "\n".join(source_lines), encoding="utf-8"
        )
        static_include = (
            f"includeFile = {args.include_prefix}/"
            f"{args.static_sources_output.name}"
        )
        run = run.replace(
            f"includeFile = {args.include_prefix}/{args.spots_output.name}",
            static_include,
        )
        run = re.sub(
            r"^d:Tf/TimelineEnd.*$\n?|"
            r"^i:Tf/NumberOfSequentialTimes.*$\n?",
            "",
            run,
            flags=re.MULTILINE,
        )
        run = re.sub(
            r'^s:Ge/BeamPosition2/Parent=.*?'
            r"^i:So/ProtonSource/NumberOfHistoriesInRun.*$\n?",
            "",
            run,
            flags=re.MULTILINE | re.DOTALL,
        )
    args.run_output.write_text(run, encoding="utf-8")
    print(
        f"Wrote {count} spots ({args.side}x{args.side}); "
        f"X=[{selected_x[0]:.9g}, {selected_x[-1]:.9g}] mm, "
        f"Y=[{selected_y[0]:.9g}, {selected_y[-1]:.9g}] mm"
    )


if __name__ == "__main__":
    main()
