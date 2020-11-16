import os


class Config:
	# def __init__(self,args):
	# 	self.a = args[0]
	# 	self.b = args[1]

	os.chdir(os.path.abspath('..'))
	os.chdir(os.path.abspath('..'))
	path = os.getcwd()
	labeled_path = os.path.join(path,'Alldata/labeled/')
	t_labeled_path = os.path.join(path, 'Alldata/labeled/')
	unlabeled_path = os.path.join(path,'Alldata/unlabeled/')

	
if __name__ == '__main__':
	args = [1,2]
	config = Config(args)
	print (config.b)

	