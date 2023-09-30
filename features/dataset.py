import shutil
import skimage
import skimage.util, skimage.io
import os
import tqdm

class Dataset:
    def __init__(self, 
            absolute_path: str, 
            ignore_dirs: list[str] = [], 
            allowed_extensions: list[str] = ["jpg", "png", "JPG", "jpeg"]):
        self.path = absolute_path
        self.ignore = ignore_dirs
        self.allowed_extensions = allowed_extensions

    def walk(self):
        for cwd, cwd_subdirs, files in os.walk(self.path):
            if (any([cwd.endswith(i) for i in self.ignore])):
                continue
            
            for f in files:
                if (not any([f.endswith(ext) for ext in self.allowed_extensions])):
                    continue 
                yield os.path.join(cwd, f) 


    def __len__(self):
        ## walk over directory
        cnt = 0
        for f in self.walk():
            cnt += 1
        return cnt        

    def __getitem__(self, i):
        if (isinstance(i, tuple) or isinstance(i, list)):
            s = sorted(i)[::-1]
            path_list = []
            for j, path in enumerate(self.walk()):
                if (s == []):
                    break
                elif (j == s[-1]):
                    path_list.append(path)
                    s.pop()
        else:
            for j, path in enumerate(self.walk()):
                if (j == i):
                    return path
        
    def load(self,
             *,
            as_gray: bool = False):
        for path in self.walk():
            img = skimage.io.imread(path, as_gray=as_gray)
            yield img

    def copy_directory_struct(self, new_path: str, no_files: bool = True):
        ignore_files = lambda directory_prefix, files: [f for f in files if os.path.isfile(os.path.join(directory_prefix, f))]

        def main_ignore(directory_prefix, paths):
            ret = []
            for f in paths: # might be directory or file
                ## TODO review this
                full = os.path.join(directory_prefix, f)
                if (any([full.endswith(d) for d in self.ignore])) \
                        or (no_files and f in ignore_files(directory_prefix, paths)):
                    ret.append(f)
            return ret

        shutil.copytree(self.path, new_path, ignore=main_ignore,
                        dirs_exist_ok=True) ## the last option implies "overwrite every time"

    def apply(self, 
            new_root: str,
            T_list: list|tuple,
            *,
            as_float: bool = False,
            warn_contrast: bool = False):

        for cwd, cwd_subdirs, files in os.walk(self.path):
            if (any([cwd.endswith(i) for i in self.ignore])):
                continue

            ## use tqdm progress bar if in leaf directory (no other subdirectories), which means the images are being created
            if cwd_subdirs == []: 
                progress_bar_iterable = tqdm(files)
                print(f"Creating folder {cwd.replace(self.path, new_root)}")
            else:
                progress_bar_iterable = files
            
            for file in progress_bar_iterable: ## use progress bar
                if not any([file.endswith(ext) for ext in self.allowed_extensions]):
                    continue
                filepath = os.path.join(cwd, file)
                img = skimage.io.imread(filepath)
                if (as_float):
                    img = skimage.utils.img_as_float(img)
                
                new_img = img
                ## apply left-to-right
                for transform in T_list:
                    new_img = transform(new_img)
                new_img = skimage.util.img_as_ubyte(new_img) ## necessary for saving
                    
                new_filepath = filepath.replace(self.path, new_root, 1)
                skimage.io.imsave(new_filepath, new_img, check_contrast=warn_contrast)

    def lazy_apply(self, 
                   T_list: list|tuple,
                   *,
                   as_float: bool = False):
        for cwd, cwd_subdirs, files in os.walk(self.path):
            if (any([cwd.endswith(i) for i in self.ignore])):
                continue

            ## use tqdm progress bar if in leaf directory (no other subdirectories), which means the images are being created

            for file in files: ## use progress bar
                if not any([file.endswith(ext) for ext in self.allowed_extensions]):
                    continue
                
                filepath = os.path.join(cwd, file)
                img = skimage.io.imread(filepath)
                if as_float:
                    img = skimage.util.img_as_float(img)
          
                ## apply left-to-right
                new_img = img
                for transform in T_list:
                    # print(transform, new_img.dtype) # TODO REMOVE
                    new_img = transform(new_img)
                
                yield new_img # lazy return (returns generator)
