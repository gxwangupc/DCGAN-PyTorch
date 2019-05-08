import argparse
import os
import subprocess
from urllib.request import Request, urlopen

def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')

def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./data', help='path to store the downloaded data')
    parser.add_argument('--category', default=None, help='category desired to be downloaded')
    args = parser.parse_args()

    categories = list_categories()
    if args.category is None:
        print('Downloading', len(categories), 'categories')
        for category in categories:
            download(args.out, category, 'train')
            download(args.out, category, 'val')
        download(args.out, '', 'test')
    else:
        if args.category == 'test':
            download(args.out, '', 'test')
        elif args.category not in categories:
            print('Error:', args.category, "doesn't exist in", 'LSUN release')
        else:
            download(args.out, args.category, 'train')
            download(args.out, args.category, 'val')


if __name__ == '__main__':
    main()
