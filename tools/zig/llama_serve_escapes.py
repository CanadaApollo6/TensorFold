"""Review minimal repros and Python source-escape boundaries, labelled by the pod."""
TOOLS = [dict(type='function', function=dict(name='edit', parameters=dict(type='object', properties=dict(
    path=dict(type='string'), edits=dict(type='array')))))]


def cases():
    values = [
        ('named-minimal', r"'\N{SNOWMAN}'"),
        ('escape-minimal', r"'\x+1'"),
        ('underscore-minimal', r"'\u0_41'"),
        ('nul-minimal', "'\0'"),
        ('edit-named', r"[{'oldText':'x','newText':'\N{SNOWMAN}'}]"),
        ('edit-invalid', r"[{'oldText':'x','newText':'\u+041'}]"),
    ]
    for name in ['SNOWMAN', 'snowman', 'LATIN CAPITAL LETTER GHA', 'latin capital letter gha', 'NULL', 'nul',
                 'CJK UNIFIED IDEOGRAPH-3400', 'CJK UNIFIED IDEOGRAPH-04E00', 'CJK UNIFIED IDEOGRAPH-323AF',
                 'cjk unified ideograph-4e00', 'CJK UNIFIED IDEOGRAPH-4e00', 'CJK UNIFIED IDEOGRAPH-004E00',
                 'CJK UNIFIED IDEOGRAPH-323B0', 'CJK COMPATIBILITY IDEOGRAPH-F900',
                 'cjk compatibility ideograph-f900', 'CJK COMPATIBILITY IDEOGRAPH-0F900',
                 'KHITAN SMALL SCRIPT CHARACTER-18B00', 'khitan small script character-18cd5',
                 'KHITAN SMALL SCRIPT CHARACTER-18CD6', 'NUSHU CHARACTER-1B170', 'nushu character-1b2fb',
                 'NUSHU CHARACTER-01B170', 'TANGUT IDEOGRAPH-17000', 'HANGUL SYLLABLE GA',
                 'HANGUL SYLLABLE HIH', 'HANGUL SYLLABLE GGWAEGS', 'hangul syllable ga', 'HANGUL SYLLABLE ga',
                 ' snowman', 'SNOWMAN ', '', 'NOT A UNICODE NAME', 'KEYCAP NUMBER SIGN']:
        values.append(('name-' + str(len(values)), "['\\N{" + name + "}']"))
    for value in [r"'\N'", r"'\N{'", r"'\N{}'", r"'\N{SNOWMAN'", r"'\N{SNOWMAN}}'",
                  r"r'\N{SNOWMAN}'", r'"\\N{SNOWMAN}"', r"'\x00'", r'"\u0000"',
                  "r'\0'", "'''\0'''", "'x' #\0", '[1,\0]', '[1] #\0', '\0[1]', '[1]\0',
                  r"['\N{NULL}', '\x00']", r"'\x41' '\u0042' '\U0001F600'", r"'\U00110000'"]:
        values.append(('boundary-' + str(len(values)), value))
    for marker, width in [('x', 2), ('u', 4), ('U', 8)]:
        for position in range(width):
            for invalid in ['+', '-', '_', 'g', ' ']:
                digits = '0' * position + invalid + '1' * (width - position - 1)
                values.append((f'hex-{marker}-{position}-{ord(invalid)}', "['\\" + marker + digits + "']"))
    return [dict(name='escape-' + name, raw='</think><tool_call><function=edit><parameter=edits>' + value +
                 '</parameter></function></tool_call>', tools=TOOLS) for name, value in values]
