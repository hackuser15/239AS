import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import sys
import os
import glob

def main():
    print("Enter the path for restructuring of folders:")
    ans = sys.stdin.readline()
    path = ans.strip('\n')
    if path.endswith(' '):
        path = path.rstrip(' ')
    print("Reorganizing folders, into the two classes expected")
    reorganize_data(path)
    delete_incompatible_files_and_load(path)

def reorganize_data(path):
    Computer_Technology = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
    Recreational_Activity_groups = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    folders = glob.glob(path + '/*')
    if len(folders) == 2:
        return
    else:
        # create `Computer_Technology` and `Recreational_Activity_groups` directories
        if not os.path.exists(path + '/' + 'comp'):
            os.makedirs(path + '/' + 'comp')
        if not os.path.exists(path + '/' + 'rec'):
            os.makedirs(path + '/' + 'rec')

        for i in Computer_Technology:
            files = glob.glob(path + '/' + i + '/*')
            for f in files:
                parts = f.split('\\')
                name = parts[len(parts) -1]
                newname = i + '_' + name
                os.rename(f, path+'/comp/'+newname)

        for i in Recreational_Activity_groups:
            files = glob.glob(path + '/' + i + '/*')
            for f in files:
                parts = f.split('\\')
                name = parts[len(parts) -1]
                newname = i + '_' + name
                os.rename(f, path+'/rec/'+newname)

def delete_incompatible_files_and_load(path = None):
    dir_path = path or 'dataset'
    print(dir_path)
    num = find_incompatible_files(dir_path)
    delete_incompatible_files(num)
    # load data
    print('Loading files into memory')
    files = sklearn.datasets.load_files(dir_path)

def delete_incompatible_files(files):
    for f in files:
        os.remove(f)

def find_incompatible_files(path):
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    files = sklearn.datasets.load_files(path)
    num = []
    for i in range(len(files.filenames)):
        try:
            count_vector.fit_transform(files.data[i:i+1])
        except UnicodeDecodeError:
            num.append(files.filenames[i])
        except ValueError:
            pass
    return num

if __name__ == '__main__':
    main()