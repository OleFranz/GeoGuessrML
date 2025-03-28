import os
import shutil

Path = os.path.dirname(__file__).replace("\\", "/")
if Path[-1] != "/":
    Path += "/"
DatasetPath = f"{Path}dataset"
DatasetCache = f"{Path}dataset-cache"

if os.path.exists(DatasetCache):
    try:
        shutil.rmtree(DatasetCache)
    except:
        pass

os.environ['HF_HOME'] = DatasetPath
os.makedirs(DatasetPath, exist_ok=True)
os.makedirs(DatasetCache, exist_ok=True)

for i in range(1000):
    Success = False
    try:
        import datasets
        Dataset = datasets.load_dataset(path="deboradum/GeoGuessr-countries",
                                        cache_dir=DatasetCache,
                                        download_config=datasets.DownloadConfig(resume_download=True, max_retries=100),
                                        verification_mode=datasets.VerificationMode.ALL_CHECKS)
        Success = True
    except:
        pass
    if Success == True:
        break

print(Dataset)