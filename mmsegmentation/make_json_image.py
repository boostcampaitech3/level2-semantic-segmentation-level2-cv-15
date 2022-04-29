import os
import json
import shutil

'''
Set json_dir
'''
DATAROOT = "./"
TRAINJSON = os.path.join(DATAROOT,"./train_all.json")  # 모든 데이터 들어있는 json 파일
TESTJSON = os.path.join(DATAROOT,"./test.json")        # test 데이터 들어있는 json 파일
'''
Redistribution image by train/valid/test 
rename img file name by img_id 
'''
def _rename_images(json_dir, image_dir):
	with open(json_dir, "r", encoding="utf8") as outfile:
		json_data = json.load(outfile)
	image_datas = json_data["images"]

	for image_data in image_datas:
		shutil.copyfile(os.path.join(image_data['file_name']), os.path.join(image_dir,f"{image_data['file_name'].split('.')[0].replace('/','')}.jpg"))

'''
Wrap func
'''
def make(json,path):
	imagePath = '../mmseg/'+path
	os.makedirs(imagePath, exist_ok=True)
	_rename_images(json,imagePath)


'''
Main
'''
def __main__():
	make(TRAINJSON, 'images')
	make(TESTJSON, 'test')


if __name__=='__main__':
	__main__()