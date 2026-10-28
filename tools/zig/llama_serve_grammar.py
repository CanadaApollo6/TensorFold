"""Seeded model-like XML grammar for the pod-labelled Nanbeige byte differential."""
import random
import unicodedata

SCHEMAS = dict(s={'type': 'string'}, i={'type': 'integer'}, n={'type': 'number'}, b={'type': 'boolean'}, o={'type': 'object'}, a={'type': 'array'}, e={'type': 'string', 'enum': ['auto', 'never']}, u={'type': ['null', 'integer']})
TOOLS = [{'type':'function', 'function':{'name':'run', 'parameters':{'type':'object', 'properties':SCHEMAS}}}]
VALUES = {
's':['echo hi', '  echo hi  ', '中文💡', '\nhi\n\n', '', 'null', '<parameter=s>inner', '<parameter=>bad', '</function>', '</tool_call>', '<parameter=s\nx>bad', 'x\x1c'],
'i':['0','-1','+7','0007','1_000',' 42 ','１２','١٢','123456789012345678901234567890', '3.0','1e2','null','NULL','nan','bad'],
'n':['0','1','1.0','1e2','1e-8','-0','-0.0','2.5','1_000',' 3.5 ','１２','١٢','nan','inf','-Infinity','1e309','null','bad'],
'b':['true','TRUE',' True ','false','False','1','0','null','NULL','bad','\ntrue\n'],
'o':['{}','{"a": 1, "b": [false, null]}',"{'a': True, 'b': None}","{'a': (1, 2)}",'[]','true','null','bad',"{1: 'x'}", "{'x': '\\x41'}", "{'x': 0x10}","{'x': -1.5}","{'x': r'\\n'}"],
'a':['[]','[1, 2.5, false, null]',"[True, None, {'x': 2}]",'(1, 2)','[1,]','null','bad','[0x10, 0b11]','[1e2, -0.0]'],
'e':['auto','never','invalid','  auto  ','null','中文'],
'u':['null','3','-2','3.5','NULL','１２']}

def generate(n=1000):
    rng=random.Random(0x8fd873cb)
    cases=[]
    for label,raw in [
        ('header-close','</think><tool_call><function=</function></tool_call>'),
        ('param-newline','</think><tool_call><function=run><parameter=s>hi\n\n</tool_call>'),
        ('function-newline','</think><tool_call><function=run><parameter=s>hi\n\n</function></tool_call>'),
        ('tool-newline','</think><tool_call><function=run><parameter=s>hi\n\n\n'),
        ('trailing-spaces','</think><tool_call><function=run><parameter=s>hi  ')]:
        cases.append(dict(name=label, raw=raw, tools=TOOLS))
    for idx in range(n):
        keys=rng.sample(list(SCHEMAS), rng.randint(1,5))
        fields=[]
        for key in keys:
            header=rng.choice([key]*12+[' '+key+' ', '\u2003'+key+'\u2003', key.upper(), '',key+'\nx','<parameter='+key,'</parameter'])
            value=rng.choice(VALUES[key])
            fields.append('<parameter='+header+'>'+rng.choice(['','\n','\n\n'])+value+rng.choice(['','\n','\n\n'])+rng.choice(['</parameter>']*8+['']))
        fname=rng.choice(['run']*20+[' run ','\u2003run\u2003','RUN','other','','</function','<function=run','ru\nn'])
        call='<tool_call>'+rng.choice(['','\n','<function=>','<function=bad\n>'])+'<function='+fname+'>'+''.join(fields)+rng.choice(['</function>']*5+[''])+rng.choice(['</tool_call>']*5+[''])
        if idx%7==0: call += '\n'+call
        if idx%11==0: call='<tool_call>broken</tool_call>'+call
        if idx%4==0: call=call[:rng.randrange(len(call)+1)]
        raw=rng.choice(['Plan.','<think>\nPlan.\n','\n  plan  \n','', '计划💡。'])+rng.choice(['</think>']*19+[''])+rng.choice(['','\n  Checking.  \n',' \t','\u2003'])+call+rng.choice(['','\n','\n\n','  ','\n  ','\nafter'])
        cases.append(dict(name='grammar-'+str(idx), raw=raw, tools=[] if idx%19==0 else TOOLS))
    return cases


def typed_cases():
    values = {k: list(v) for k, v in VALUES.items()}
    values['i'] += ['\u2003+١_٢\u00a0', '-１２_３', '\x1c1\x1c', '1__0', '_1', '1_', '0x10']
    values['i'] += [chr(c) + chr(c + 1) for c in range(0x110000)
                    if unicodedata.category(chr(c)) == 'Nd' and unicodedata.digit(chr(c)) == 0]
    values['n'] += ['1.0e16', '1.5e-07', '1e16', '  -INFINITY  ', '+NaN', '1e308', '١.٥e٢',
                    '0.100000000000000005', '1_.0', '1e_2', '.5', '5.', '1_0.5_0']
    values['o'] += ["{1e309: 1, -1e309: 2}", "{True: 1, 1: 2, False: 3, 0.0: 4}", "{1: 'x', '1': 'y'}", "{1.0: 'x', 1: 'y'}", 'TRUE', 'False', 'None', "{'x': +1, 'y': -2.5}", "{'x': 1, 'x': 2}",
                    '{"x": 1, "x": 2}', "{'x': 0xffffffffffffffffffffffffffffffffffffffffffff}",
                    "{'x': '\\u4e2d', 'y': '\\x41'}", "{'x': [1, (2, 3), None]}"]
    values['a'] += ['1, 2', '[1,]\n#comment', '[1,]\n  \n', '[1,]\n\f', '[-(1), +(2.0)]', '[1,]\n ', ' (1, 2) ', '[+1, -2, (3,), True]', '[--1]', '[1__0]',
                    "['a' 'b', r'\\n']", '[1,]\n', '[1,]\u00a0', '[1,]\n  ']
    schemas = dict(SCHEMAS, z={'type': 'null'}, v={'type': ['null']}, w={'type': ['null', 'number']},
                   x={'type': ['string', 'integer']}, j={'anyOf': [{'type': 'number'}]},
                   text={'type': 'text'}, integer={'type': 'int'}, boolean={'type': 'bool'})
    values.update(z=['null', 'NULL', 'None', 'true', '1', 'bad'], v=['null', '3'], w=['1e2', 'null'],
                  x=['1', 'null'], j=['1', 'null'], text=['null', 'NULL'], integer=['١'], boolean=['TRUE'])
    tools = [{'type': 'function', 'function': {'name': 'run', 'parameters': {'properties': schemas}}}]
    out = []
    for key, items in values.items():
        for i, value in enumerate(items):
            raw = '</think><tool_call><function=run><parameter=' + key + '>' + value + '</parameter></function></tool_call>'
            out.append(dict(name=f'typed-{key}-{i}', raw=raw, tools=tools))
    for name, raw in [('lfm-prose', '</think><|tool_call_start|>[run()]<|tool_call_end|>'),
                      ('lfm-argument', '</think><tool_call><function=run><parameter=s><|tool_call_start|>[run()]<|tool_call_end|>'),
                      ('duplicate-parameters', '</think><tool_call><function=run><parameter=i>1<parameter=s>中<parameter=i>2')]:
        out.append(dict(name=name, raw=raw, tools=tools))
    return out
