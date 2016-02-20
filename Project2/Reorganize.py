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
	#init()

	# get the dataset
	print("Where is the dataset?")
	print('warning: files might get deleted if they are incompatible with utf8')
	ans = sys.stdin.readline()
	# remove any newlines or spaces at the end of the input
	path = ans.strip('\n')
	if path.endswith(' '):
		path = path.rstrip(' ')

	# preprocess data into two folders instead of 6
	print("Reorganizing folders, into two classes")
	reorganize_dataset(path)
	print('\n\n')
	# do the main test
	main_test(path)
#    dir_path = sys.path or 'dataset'
	#remove_incompatible_files(dir_path)


def reorganize_dataset(path):
	Computer_Technology = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
	Recreational_Activity_groups = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

	folders = glob.glob(path + '/*')
	if len(folders) == 2:
		return
	else:
		# create `likes` and `dislikes` directories
		if not os.path.exists(path + '/' + 'comp'):
			os.makedirs(path + '/' + 'comp')
		if not os.path.exists(path + '/' + 'rec'):
			os.makedirs(path + '/' + 'rec')

		for like in Computer_Technology:
			files = glob.glob(path + '/' + like + '/*')
			for f in files:
				parts = f.split('\\')
				name = parts[len(parts) -1]
				newname = like + '_' + name
				os.rename(f, path+'/comp/'+newname)
			#os.rmdir(path + '/' + like)

		for like in Recreational_Activity_groups:
			files = glob.glob(path + '/' + like + '/*')
			for f in files:
				parts = f.split('\\')
				name = parts[len(parts) -1]
				newname = like + '_' + name
				os.rename(f, path+'/rec/'+newname)
			#os.rmdir(path + '/' + like)

def main_test(path = None):
    dir_path = path or 'dataset'
    print(dir_path)
    #remove_incompatible_files(dir_path)
    print('\n\n')
    # load data
    print('Loading files into memory')
    files = sklearn.datasets.load_files(dir_path)
    print(files.target)
    print(len(files.data))
    num = find_incompatible_files(dir_path)
    delete_incompatible_files(num)

def delete_incompatible_files(files):
	import os
	for f in files:
		print("deleting file:")
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