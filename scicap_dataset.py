import os
from PIL import Image
import json
import csv
from torch.utils.data import Dataset

#from lavis.models import load_model_and_preprocess


class SciCapDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 transform = None,
                 train = True,               # 学習用データなのか
                 train_include_val = True,         # データにvalデータを含めるか
                 include_subfig = False,     # データにsubfigデータを含めるか
                 tokenizer=None,
                 image_processor=None,
                 #text_tokenizer=None
                 ):
        
        self.path = dataset_path
        self.transform = transform
        self.train = train
        self.train_include_val = train_include_val
        self.include_subfig = include_subfig
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.abst_file = os.path.join(self.path, 'id_abstract.csv')
        self.image_filenames = self._load_image_filenames()

        #self.abstract = csv.DictReader(self.abst_file)
        #
        #with open(self.abst_file, newline='') as csvfile:
        #    self.abstract = csv.DictReader(csvfile)
        #
        #self.abst_list = {}
        #for row in self.abstract:
        #    self.abst_dict[row['id']] = row['abstract']
        
        self.abst_dict = self.load_data()
        
        print('end of SciCapDataset __init__')


    def load_data(self):
        abst_dict = {}
        with open(self.abst_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                abst_dict[row['id']] = row['abstract']
        return abst_dict

    def _load_image_filenames(self):
        image_filenames = []

        if self.train == True:
            file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "train")
            filenames = os.listdir(file_path)
            image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.include_subfig == True:
                file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "train")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.train_include_val == True:
                file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "val")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

                if self.include_subfig == True:
                    file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "val")
                    filenames = os.listdir(file_path)
                    image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])
        elif self.train == False:
            file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "test")
            filenames = os.listdir(file_path)
            image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.include_subfig == True:
                file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "test")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

            if self.train_include_val == False:
                file_path = os.path.join(self.path, "SciCap-No-Subfig-Img", "val")
                filenames = os.listdir(file_path)
                image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])

                if self.include_subfig == False:
                    file_path = os.path.join(self.path, "SciCap-Yes-Subfig-Img", "val")
                    filenames = os.listdir(file_path)
                    image_filenames.extend([os.path.join(file_path, filename) for filename in filenames])
        elif self.train == 'ALL':
            data_paths = [os.path.join(self.path, 'SciCap-No-Subfig-Img/test'),
                          os.path.join(self.path, 'SciCap-No-Subfig-Img/train'),
                          os.path.join(self.path, 'SciCap-No-Subfig-Img/val'),
                          os.path.join(self.path, 'SciCap-Yes-Subfig-Img/test'),
                          os.path.join(self.path, 'SciCap-Yes-Subfig-Img/train'),
                          os.path.join(self.path, 'SciCap-Yes-Subfig-Img/val'),
                          ]
            for data_path in data_paths:
                filenames = os.listdir(data_path)
                image_filenames.extend([os.path.join(data_path, filename) for filename in filenames])
            
        return image_filenames

    def _change_extension(self, filename, new_extension):
        base_name, _ = os.path.splitext(filename)
        return f"{base_name}.{new_extension}"
    
    # "v" より前の部分を取り出す関数
    def extract_version(self,input_string):
        parts = input_string.split('v')
        if len(parts) > 1:
            return parts[0]
        else:
            return None


    def __len__(self):
        return len(self.image_filenames)

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result  

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        img = Image.open(img_path).convert('RGB')
        
        img = self.expand2square(pil_img=img, background_color=(255,255,255))
        
        if self.image_processor:
            img = self.image_processor(img)
            

        file_name_with_extension = os.path.basename(img_path)
        file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
        dir_path = os.path.dirname(img_path)  # ファイルパスからディレクトリパスを取得
        directory_name = os.path.basename(dir_path)
        cap_path = os.path.join(self.path, 'SciCap-Caption-All', directory_name, file_name_without_extension+'.json')
 
        with open(cap_path, 'r') as json_file:
            data = json.load(json_file)
            caption = data.get("0-originally-extracted")  # キーが"0-originally-extracted"の値を取得
            caption = 'Description: ' + caption
            paper_id = data.get("paper-ID")
        
        target_id = self.extract_version(paper_id)
        

        #with open(self.abst_file, 'r') as csv_file:
        #    reader = csv.DictReader(csv_file)
        #    for row in reader:
        #        if row['id'] == target_id:
        #            abstract = row['abstract']

        # desired_idに対応するabstractを取得
        if target_id in self.abst_dict:
            abstract = self.abst_dict[target_id]
            #print(abstract)
        else:
            print(f"id {target_id} は見つかりませんでした。")
        
        if self.tokenizer:
            input_prompt = f"{abstract}<|endofchunk|><image>{caption}<|endofchunk|>{self.tokenizer.eos_token}"
        else:
            input_prompt = f"{abstract}<|endofchunk|><image>{caption}<|endofchunk|>self.tokenizer.eos_token"
            
        return img, input_prompt


if __name__ == '__main__':

    dataset = SciCapDataset(dataset_path = '/taiga/Datasets/scicap_data', 
                       transform = None,
                       train = 'ALL',               # 学習用データなのか
                       train_include_val = True,         # データにvalデータを含めるか
                       include_subfig = True,     # データにsubfigデータを含めるか
                       image_processor=None,
                       tokenizer=None
                       )
    

    img, input_prompt = dataset[0]

    print('img : ', img)
    print('input_prompt: ', input_prompt)
