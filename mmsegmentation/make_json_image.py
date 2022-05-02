import os
import json
import shutil


DATAROOT = "./"
TRAINJSON = os.path.join(DATAROOT,"./train_all.json")  # 모든 데이터 들어있는 json 파일
TESTJSON = os.path.join(DATAROOT,"./test.json")        # test 데이터 들어있는 json 파일


def _rename_images(json_dir, image_dir, id=False):
	with open(json_dir, "r", encoding="utf8") as outfile:
		json_data = json.load(outfile)
	image_datas = json_data["images"]

	if id:
		for image_data in image_datas:
			shutil.copyfile(os.path.join(image_data['file_name']), os.path.join(image_dir,f"{image_data['id']:04}_{image_data['file_name'].split('.')[0].replace('/','')}.jpg"))
	else:
		for image_data in image_datas:
			shutil.copyfile(os.path.join(image_data['file_name']), os.path.join(image_dir,f"{image_data['file_name'].split('.')[0].replace('/','')}.jpg"))


def make(json, path, id=False):
	imagePath = '../mmseg/'+path
	os.makedirs(imagePath, exist_ok=True)
	_rename_images(json, imagePath, id)


def __main__():
	make(TRAINJSON, 'images', id=False)
	make(TESTJSON, 'test', id=True)


if __name__=='__main__':
	__main__()