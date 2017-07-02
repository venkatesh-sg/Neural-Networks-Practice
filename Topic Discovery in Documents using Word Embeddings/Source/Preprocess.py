import os

# Directory of raw files
Directory="PMC_corpus_lite_train/"
for subdir, dirs, files in os.walk(Dir):
    for file in files:
        f = open(os.path.join(subdir,file),"r",encoding="utf8")
        lines = f.readlines()
        f.close()
        f=open("Data/"+file,"w",encoding="utf8")
        for line in lines:
            if (line !="==== Refs\n"):
                f.write(line)
            else:
                break

        
