"""Generate and exhaustively check Python 3.12's Unicode 15 literal-name database."""
import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unicodedata as ud
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / 'zig/src/server'
DATA = SERVER / 'fixtures/llama/unicode-15.names.gz'
SOURCES = {
    'NameAliases.txt': '3e39509e8fae3e5d50ba73759d0b97194501d14a9c63107a6372a46b38be18e8',
    'NamedSequences.txt': '1d5202155f14841973aa540b1625f4befbde185ac77ce5aceaaaa0501a68bd66',
}
PREFIXES = ['CJK UNIFIED IDEOGRAPH-', 'KHITAN SMALL SCRIPT CHARACTER-', 'NUSHU CHARACTER-']
LEADS = 'G GG N D DD R M B BB S SS _ J JJ C K T P H'.replace('_', '').split(' ')
VOWELS = 'A AE YA YAE EO E YEO YE O WA WAE OE YO U WEO WE WI YU EU YI I'.split()
TAILS = ('_ G GG GS N NJ NH D L LG LM LB LS LT LP LH M B BS S SS NG J C K T P H'
         .replace('_', '').split(' '))


def source(name, directory):
    path = directory / name if directory else None
    raw = path.read_bytes() if path else urlopen('https://www.unicode.org/Public/15.0.0/ucd/' + name).read()
    assert hashlib.sha256(raw).hexdigest() == SOURCES[name], name
    return [line.split(';') for line in raw.decode().splitlines() if line and not line.startswith('#')]


def literal(name):
    try:
        return ord(ast.literal_eval("'\\N{" + name + "}'"))
    except SyntaxError:
        return None


def database(directory):
    named = {ud.name(chr(cp)): cp for cp in range(0x110000) if ud.name(chr(cp), None)}
    aliases = {name: int(cp, 16) for cp, name, _ in source('NameAliases.txt', directory)}
    for name, cp in aliases.items():
        assert ud.lookup(name) == chr(cp)
    sequences = [name for name, _ in source('NamedSequences.txt', directory)]
    for name in sequences:
        assert len(ud.lookup(name)) > 1
    families = []
    for prefix in PREFIXES:
        points = sorted(cp for name, cp in named.items() if name.startswith(prefix))
        ranges = []
        for cp in points:
            if ranges and cp == ranges[-1][1] + 1:
                ranges[-1][1] = cp
            else:
                ranges.append([cp, cp])
        families.append(dict(prefix=prefix, ranges=ranges, insensitive=literal((prefix + f'{points[0]:X}').lower()) is not None))
    explicit = {name: cp for name, cp in named.items()
                if not name.startswith(tuple(PREFIXES + ['HANGUL SYLLABLE ']))}
    explicit.update(aliases)
    return named, aliases, sequences, explicit, families


def packed(entries):
    blob, offsets = bytearray(), bytearray()
    for name, cp in sorted(entries.items()):
        offsets += struct.pack('<I', len(blob))
        blob += cp.to_bytes(3, 'little') + name.encode('ascii') + b'\0'
    raw = struct.pack('<I', len(entries)) + offsets + blob
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, filename='', mode='wb', mtime=0) as out:
        out.write(raw)
    return compressed.getvalue()


def unpack(data):
    raw = gzip.decompress(data)
    count, = struct.unpack_from('<I', raw)
    entries = {}
    for i in range(count):
        offset, = struct.unpack_from('<I', raw, 4 + i * 4)
        at = 4 + count * 4 + offset
        cp = int.from_bytes(raw[at:at + 3], 'little')
        entries[raw[at + 3:raw.index(b'\0', at + 3)].decode()] = cp
    return entries


def metadata(families):
    lines = ['pub const Family = struct {', '    prefix: []const u8,', '    insensitive: bool,',
             '    ranges: []const [2]u21,', '};', '', 'pub const families = [_]Family{']
    for family in families:
        lines += ['    .{', f'        .prefix = "{family["prefix"]}",',
                  f'        .insensitive = {str(family["insensitive"]).lower()},', '        .ranges = &.{']
        lines += [f'            .{{ 0x{lo:x}, 0x{hi:x} }},' for lo, hi in family['ranges']]
        lines += ['        },', '    },']
    lines += ['};', '']
    return '\n'.join(lines)


def lookup(name, entries, families):
    if name.startswith('HANGUL SYLLABLE '):
        rest, parts = name[16:], []
        for group in [LEADS, VOWELS, TAILS]:
            matches = [(len(s), i) for i, s in enumerate(group) if rest.startswith(s)]
            if not matches:
                return None
            size, i = max(matches)
            rest = rest[size:]
            parts.append(i)
        return None if rest else 0xac00 + (parts[0] * 21 + parts[1]) * 28 + parts[2]
    for family in families:
        candidate = name.upper() if family['insensitive'] else name
        if not candidate.startswith(family['prefix']):
            continue
        suffix = candidate[len(family['prefix']):]
        if len(suffix) not in ([5] if family['insensitive'] else [4, 5]):
            return None
        if any(c not in '0123456789ABCDEF' for c in suffix):
            return None
        cp = int(suffix, 16)
        return cp if any(lo <= cp <= hi for lo, hi in family['ranges']) else None
    return entries.get(name.upper())


def candidates(named, aliases, sequences, families):
    for kind, entries in [('named', named), ('alias', aliases)]:
        for name in entries:
            yield kind, name
            yield kind, name.lower()
    for name in sequences:
        yield 'sequence', name
        yield 'sequence', name.lower()
    for name in list(named)[::101] + list(aliases):
        for variant in [' ' + name, name + ' ', name.replace(' ', '  '), name + '-UNKNOWN']:
            yield 'negative', variant
    for family in families:
        for lo, hi in family['ranges']:
            for cp in [lo - 1, lo, hi, hi + 1]:
                for digits in [f'{cp:X}', f'{cp:05X}', f'{cp:06X}', '+' + f'{cp:X}', f'{cp:X}_']:
                    for prefix in [family['prefix'], family['prefix'].lower()]:
                        yield 'boundary', prefix + digits
    for name in ['', 'snowman ', ' snowman', 'SNOWMAN\t', 'SNOWMAN\n', 'ＳＮＯＷＭＡＮ',
                 'TANGUT IDEOGRAPH-17000', 'HANGUL SYLLABLE', 'HANGUL SYLLABLE GAX',
                 'HANGUL SYLLABLE GA ', 'HANGUL SYLLABLE ga', 'CJK UNIFIED IDEOGRAPH-4e00']:
        yield 'negative', name


def verify(data, named, aliases, sequences, families, zig):
    entries = unpack(data)
    counts = {}
    with tempfile.TemporaryDirectory(prefix='nanbeige-unicode-') as temporary:
        corpus = Path(temporary) / 'names.tsv'
        with corpus.open('w') as out:
            for kind, name in candidates(named, aliases, sequences, families):
                expected = literal(name)
                assert lookup(name, entries, families) == expected, (kind, name, expected)
                counts[kind] = counts.get(kind, 0) + 1
                if '\n' not in name:
                    out.write(f'{expected if expected is not None else -1}\t{name}\n')
        if zig:
            subprocess.run([str(zig), 'test', '-lc', '-O', 'ReleaseSafe', '--dep', 'names',
                            '-Mroot=' + str(ROOT / 'tools/zig/llama_unicode_check.zig'),
                            '-Mnames=' + str(SERVER / 'nanbeige_unicode.zig')], check=True,
                           env=dict(os.environ, UNICODE_NAMES_INPUT=str(corpus), CUDA_VISIBLE_DEVICES=''))
    print(json.dumps(dict(python=sys.version.split()[0], unicode=ud.unidata_version,
                          explicit_and_aliases=len(entries), named_codepoints=len(named), aliases=len(aliases),
                          compressed_bytes=len(data), checks=counts, mismatches=0, zig_checked=bool(zig)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ucd-dir', type=Path)
    parser.add_argument('--zig', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12) and ud.unidata_version == '15.0.0'
    named, aliases, sequences, entries, families = database(args.ucd_dir)
    data = packed(entries)
    assert len(data) < 300_000
    meta = metadata(families)
    path = SERVER / 'nanbeige_unicode_data.zig'
    if args.check:
        assert DATA.read_bytes() == data and path.read_text() == meta, 'regenerate Unicode data'
    else:
        DATA.write_bytes(data)
        path.write_text(meta)
    verify(data, named, aliases, sequences, families, args.zig)


if __name__ == '__main__':
    main()
