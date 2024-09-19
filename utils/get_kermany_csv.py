# import subprocess
#
# ps = subprocess.Popen(['unzip', '-l', '/scratch/avram/Kermany/kermany2018.zip'], stdout=subprocess.PIPE)
# ps = subprocess.Popen(['grep', 'oct2017/OCT2017 '], stdout=subprocess.PIPE)
# output = subprocess.check_output(['grep', '-v', 'DS_Store'], stdin=ps.stdout).decode()
# lines = output.split('\n')
# x = subprocess.check_output(['unzip', '-l', '/scratch/avram/Kermany/kermany2018.zip']).decode()
# x.split('\n')[:10]

def get_meta(split, mock,
             diseases={'CNV': '1,0,0,0',
                       'DME': '0,1,0,0',
                       'DRUSEN': '0,0,1,0',
                       'NORMAL': '0,0,0,1'},
             data_dir='/data1/Ophthalmology/OCT/Kermany'):
    res = ''
    with open(f'{data_dir}/{split}.txt') as f:
        lines = [line.rstrip() for line in f.readlines() if 'jpeg' in line]
    for i, line in enumerate(lines):
        disease = line.split('-')[0]
        res += f'{line},/{split}/{disease}/{line},{diseases[disease]}\n'
        if mock and i ==1000:
            break
    return res


mock = ''
mock = 'mock_'  # uncomment to create a mock dataset
with open(f'/data1/Ophthalmology/OCT/Kermany/{mock}kermany.csv', 'w') as f:
    f.write('F_name,path,Normal,DME,CNV,Drusen\n')  # TODO: remove F_name
    f.write(get_meta('train', mock))
    f.write(get_meta('test', mock))
