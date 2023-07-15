import os, tarfile, torch, shutil

def N_SFS2CM(normal):
    temp = torch.zeros_like(normal)
    temp[:, 0, :, :] = normal[:, 1, :, :]
    temp[:, 1, :, :] = -normal[:, 0, :, :]
    temp[:, 2, :, :] = normal[:, 2, :, :]
    return temp

def get_normal_255(normal):
    new_normal = normal * 128 + 128
    new_normal = new_normal.clamp(0, 255) / 255
    return new_normal

def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_targz(output_filename, source_dir):
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False


def CopyFiles(srcfile,dstpath):                      
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       
        shutil.copy(srcfile, dstpath + fname)         
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))
