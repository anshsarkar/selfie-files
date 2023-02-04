from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.output = "/dbfs/FileStore/ansh_sarkar/training_face_match_data/trained_models/2022_06_01_to_2022_09_01"
config.embedding_size = 512
config.save_all_states = True
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/dbfs/FileStore/ansh_sarkar/training_face_match_data/images/2022_06_01_to_2022_09_01/processed"
config.num_classes = 247226
config.num_image = 564789
config.num_epoch = 25
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
