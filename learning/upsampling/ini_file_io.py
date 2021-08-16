#!/usr/bin/python
import configparser

def load_train_ini(ini_file):
    # initialize
    cf = configparser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    param_sections = []

    s = cf.sections()
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(phase         = cf.get(s[d], "phase"),
                          batch_size    = cf.getint(s[d], "batch_size"),
                          inputI_width_size   = cf.getfloat(s[d], "inputI_width_size"),
                          inputI_height_size   = cf.getfloat(s[d], "inputI_height_size"),
                          r    = cf.getint(s[d], "r"),
                          factor    = cf.getint(s[d], "factor"),
                          niters    = cf.getint(s[d], "niters"),
                          inputI_chn    = cf.getint(s[d], "inputI_chn"),
                          output_chn    = cf.getint(s[d], "output_chn"),
                          ImagePath = cf.get(s[d], "ImagePath"),
                          DepthPath = cf.get(s[d], "DepthPath"),
                          chkpoint_dir  = cf.get(s[d], "chkpoint_dir"),
                          result_dir  = cf.get(s[d], "result_dir"),
                          learning_rate = cf.getfloat(s[d], "learning_rate"),
                          epoch         = cf.getint(s[d], "epoch"),
                          labeling_dir  = cf.get(s[d], "labeling_dir"))
        # add to list
        param_sections.append(level_dict)

    return param_sections
