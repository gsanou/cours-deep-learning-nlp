import json,sys,base64,os
nb=json.load(open(sys.argv[1])); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
k=0
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    for o in c['outputs']:
        if o['output_type']=='stream': print(f"--- cell {i} stdout:\n"+''.join(o['text'])[:1500])
        elif 'image/png' in o.get('data',{}):
            fn=f"{out}/c{i:03d}_{k}.png"; open(fn,'wb').write(base64.b64decode(o['data']['image/png'])); k+=1; print(f"--- cell {i} fig -> {fn}")
