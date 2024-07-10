import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import wandb
import geopandas as gpd
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
# from fvcore.nn import FlopCountAnalysis
import albumentations as A
from affine import Affine

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot,
                                          CustomVectorOutputConfig, FeatureMapVisualization)
from src.features.dataloaders import (cities, show_windows, buil_create_full_image,ensure_tuple,
                                  create_datasets, senitnel_create_full_image, vis_sent, vis_build,
                                  MergeDataset, create_scenes_for_city, PolygonWindowGeoDataset)
from rastervision.core.box import Box
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data import (ClassConfig, XarraySource, Scene,RasterioCRSTransformer,
                                    ClassInferenceTransformer, SemanticSegmentationDiscreteLabels)
from rastervision.pytorch_learner.dataset.transform import (TransformType, TF_TYPE_TO_TF_FUNC)
from rastervision.pytorch_learner import (SemanticSegmentationSlidingWindowGeoDataset,
                                          SlidingWindowGeoDataset)
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from rastervision.core.data.label_store import (SemanticSegmentationLabelStore)

# Check if MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

# Santo Domingo
sentinel_sceneSD, buildings_sceneSD = create_scenes_for_city('SantoDomingoDOM', cities['SantoDomingoDOM'], class_config)
sentinelGeoDataset_SD = PolygonWindowGeoDataset(sentinel_sceneSD, window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SD = PolygonWindowGeoDataset(buildings_sceneSD, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_SD, sent_val_ds_SD, sent_test_ds_SD = sentinelGeoDataset_SD.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_SD, buil_val_ds_SD, buil_test_ds_SD = buildGeoDataset_SD.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

img_full = senitnel_create_full_image(sentinelGeoDataset_SD.scene.label_source)
train_windows = sent_train_ds_SD.windows
val_windows = sent_val_ds_SD.windows
test_windows = sent_test_ds_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Santo Domingo - Sliding windows (Train in blue, Val in red, Test in green)')

img_full = senitnel_create_full_image(buildGeoDataset_SD.scene.label_source)
train_windows = buil_train_ds_SD.windows
val_windows = buil_val_ds_SD.windows
test_windows = buil_test_ds_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

train_datasetSD = MergeDataset(sent_train_ds_SD, buil_train_ds_SD)
val_datasetSD = MergeDataset(sent_val_ds_SD, buil_val_ds_SD)
len(train_datasetSD)

x, y = vis_sent.get_batch(sent_train_ds_SD, 10)
vis_sent.plot_batch(x, y, show=True)

# Guatemala City
sentinel_sceneGC, buildings_sceneGC = create_scenes_for_city('GuatemalaCity', cities['GuatemalaCity'], class_config)
sentinelGeoDataset_GC = PolygonWindowGeoDataset(sentinel_sceneGC, window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_GC = PolygonWindowGeoDataset(buildings_sceneGC, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_GC, sent_val_ds_GC, sent_test_ds_GC = sentinelGeoDataset_GC.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_GC, buil_val_ds_GC, buil_test_ds_GC = buildGeoDataset_GC.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

train_datasetGC = MergeDataset(sent_train_ds_GC, buil_train_ds_GC)
val_datasetGC = MergeDataset(sent_val_ds_GC, buil_val_ds_GC)
len(train_datasetGC)
img_full = senitnel_create_full_image(buil_train_ds_GC.scene.label_source)
train_windows = buil_train_ds_GC.windows
val_windows = buil_val_ds_GC.windows
test_windows = buil_test_ds_GC.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Guatemala City - Sliding windows (Train in blue, Val in red, Test in green)')

x, y = vis_sent.get_batch(sent_train_ds_GC, 29)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_GC, 5)
vis_build.plot_batch(x, y, show=True)

# TegucigalpaHND
sentinel_sceneTG, buildings_sceneTG = create_scenes_for_city('TegucigalpaHND', cities['TegucigalpaHND'], class_config)
sentinelGeoDataset_TG = PolygonWindowGeoDataset(sentinel_sceneTG, window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_TG = PolygonWindowGeoDataset(buildings_sceneTG, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_TG, sent_val_ds_TG, sent_test_ds_TG = sentinelGeoDataset_TG.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_TG, buil_val_ds_TG, buil_test_ds_TG = buildGeoDataset_TG.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

train_datasetTG = MergeDataset(sent_train_ds_TG, buil_train_ds_TG)
val_datasetTG = MergeDataset(sent_val_ds_TG, buil_val_ds_TG)
len(train_datasetTG)
img_full = senitnel_create_full_image(sentinelGeoDataset_TG.scene.label_source)
train_windows = sent_train_ds_TG.windows
val_windows = sent_val_ds_TG.windows
test_windows = sent_test_ds_TG.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='TegucigalpaHND - Sliding windows (Train in blue, Val in red, Test in green)')

x, y = vis_sent.get_batch(sentinelGeoDataset_TG, 5)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_TG, 5)
vis_build.plot_batch(x, y, show=True)

# Managua
sentinel_sceneMN, buildings_sceneMN = create_scenes_for_city('Managua', cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(sentinel_sceneMN, window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_MN = PolygonWindowGeoDataset(buildings_sceneMN, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_MN, sent_val_ds_MN, sent_test_ds_MN = sentinelGeoDataset_MN.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_MN, buil_val_ds_MN, buil_test_ds_MN = buildGeoDataset_MN.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

train_datasetMN = MergeDataset(sent_train_ds_MN, buil_train_ds_MN)
val_datasetMN = MergeDataset(sent_val_ds_MN, buil_val_ds_MN)
len(train_datasetMN)
img_full = senitnel_create_full_image(sentinelGeoDataset_MN.scene.label_source)
train_windows = sent_train_ds_MN.windows
val_windows = sent_val_ds_MN.windows
test_windows = sent_test_ds_MN.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Managua Sliding windows (Train in blue, Val in red, Test in green)')

x, y = vis_sent.get_batch(sent_train_ds_MN, 5)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_MN, 5)
vis_build.plot_batch(x, y, show=True)

# Panama
sentinel_scenePN, buildings_scenePN = create_scenes_for_city('Panama', cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN,window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_PN = PolygonWindowGeoDataset(buildings_scenePN,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_PN, sent_val_ds_PN, sent_test_ds_PN = sentinelGeoDataset_PN.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_PN, buil_val_ds_PN, buil_test_ds_PN = buildGeoDataset_PN.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

train_datasetPN = MergeDataset(sent_train_ds_PN, buil_train_ds_PN)
val_datasetPN = MergeDataset(sent_val_ds_PN, buil_val_ds_PN)
len(train_datasetPN)
img_full = senitnel_create_full_image(sentinelGeoDataset_PN.scene.label_source)
train_windows = sent_train_ds_PN.windows
val_windows = sent_val_ds_PN.windows
test_windows = sent_test_ds_PN.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Panama Sliding windows (Train in blue, Val in red, Test in green)')

x, y = vis_sent.get_batch(sent_train_ds_PN, 14)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_PN, 5)
vis_build.plot_batch(x, y, show=True)

# San Salvador
sentinel_sceneSS, buildings_sceneSS = create_scenes_for_city('SanSalvador_PS', cities['SanSalvador_PS'], class_config)
sentinelGeoDataset_SS = PolygonWindowGeoDataset(sentinel_sceneSS,window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SS = PolygonWindowGeoDataset(buildings_sceneSS,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_SS, sent_val_ds_SS, sent_test_ds_SS = sentinelGeoDataset_SS.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_SS, buil_val_ds_SS, buil_test_ds_SS = buildGeoDataset_SS.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
train_datasetSS = MergeDataset(sent_train_ds_SS, buil_train_ds_SS)
val_datasetSS = MergeDataset(sent_val_ds_SS, buil_val_ds_SS)
len(train_datasetSS)
img_full = senitnel_create_full_image(sentinelGeoDataset_SS.scene.label_source)
train_windows = sent_train_ds_SS.windows
val_windows = sent_val_ds_SS.windows
test_windows = sent_test_ds_SS.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='SanSalvador Sliding windows')

x, y = vis_sent.get_batch(sent_train_ds_SS, 14)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_SS, 5)
vis_build.plot_batch(x, y, show=True)

# SanJoseCRI
sentinel_sceneSJ, buildings_sceneSJ = create_scenes_for_city('SanJoseCRI', cities['SanJoseCRI'], class_config)
sentinelGeoDataset_SJ = PolygonWindowGeoDataset(sentinel_sceneSJ,window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SJ = PolygonWindowGeoDataset(buildings_sceneSJ,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_SJ, sent_val_ds_SJ, sent_test_ds_SJ = sentinelGeoDataset_SJ.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_SJ, buil_val_ds_SJ, buil_test_ds_SJ = buildGeoDataset_SJ.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
train_datasetSJ = MergeDataset(sent_train_ds_SJ, buil_train_ds_SJ)
val_datasetSJ = MergeDataset(sent_val_ds_SJ, buil_val_ds_SJ)

img_full = senitnel_create_full_image(sentinelGeoDataset_SJ.scene.label_source)
train_windows = sent_train_ds_SJ.windows
val_windows = sent_val_ds_SJ.windows
test_windows = sent_test_ds_SJ.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='SanSalvador Sliding windows')

x, y = vis_sent.get_batch(sentinelGeoDataset_SJ, 5)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_SJ, 5)
vis_build.plot_batch(x, y, show=True)

# BelizeCity
sentinel_sceneBL, buildings_sceneBL = create_scenes_for_city('BelizeCity', cities['BelizeCity'], class_config)
sentinelGeoDataset_BL = PolygonWindowGeoDataset(sentinel_sceneBL,window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_BL = PolygonWindowGeoDataset(buildings_sceneBL,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
sent_train_ds_BL, sent_val_ds_BL, sent_test_ds_BL = sentinelGeoDataset_BL.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_BL, buil_val_ds_BL, buil_test_ds_BL = buildGeoDataset_BL.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
train_datasetBL = MergeDataset(sent_train_ds_BL, buil_train_ds_BL)
val_datasetBL = MergeDataset(sent_val_ds_BL, buil_val_ds_BL)

img_full = senitnel_create_full_image(sentinelGeoDataset_BL.scene.label_source)
train_windows = sent_train_ds_BL.windows
val_windows = sent_val_ds_BL.windows
test_windows = sent_test_ds_BL.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Belize City Sliding windows')

x, y = vis_sent.get_batch(sentinelGeoDataset_BL, 4)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_BL, 4)
vis_build.plot_batch(x, y, show=True)

# Belmopan - data mostly rural exluding from training
sentinel_sceneBM, buildings_sceneBM = create_scenes_for_city('Belmopan', cities['Belmopan'], class_config, resolution=5)
sentinelGeoDataset_BM = PolygonWindowGeoDataset(sentinel_sceneBM, window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_BM = PolygonWindowGeoDataset(buildings_sceneBM, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

sent_train_ds_BM, sent_val_ds_BM, sent_test_ds_BM = sentinelGeoDataset_BM.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
buil_train_ds_BM, buil_val_ds_BM, buil_test_ds_BM = buildGeoDataset_BM.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)
train_datasetBL = MergeDataset(sent_train_ds_BM, buil_train_ds_BM)
val_datasetBL = MergeDataset(sent_val_ds_BM, buil_val_ds_BM)

img_full = senitnel_create_full_image(sentinelGeoDataset_BM.scene.label_source)
train_windows = sent_train_ds_BM.windows
val_windows = sent_val_ds_BM.windows
test_windows = sent_test_ds_BM.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Belmopan Sliding windows')

img_full = senitnel_create_full_image(buildGeoDataset_BM.scene.label_source)
train_windows = buil_train_ds_BM.windows
val_windows = buil_val_ds_BM.windows
test_windows = buil_test_ds_BM.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Belmopan City Sliding windows')

x, y = vis_sent.get_batch(sentinelGeoDataset_BM, 2)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildGeoDataset_BM, 2)
vis_build.plot_batch(x, y, show=True)

train_dataset = ConcatDataset([train_datasetSD, train_datasetGC, train_datasetTG, train_datasetMN, train_datasetPN, train_datasetSS, train_datasetBL]) #,  train_datasetSJ, train_datasetBM
val_dataset = ConcatDataset([val_datasetSD, val_datasetGC, val_datasetTG, val_datasetMN, val_datasetPN, val_datasetSS, val_datasetBL])
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")

# Initialize the data module
batch_size = 18
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
data_module = MultiModalDataModule(train_loader, val_loader)

# Train the model 
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'all',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'labels_size': 256,
    'buil_channels': 128,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'gamma': 0.8,
    'sched_step_size': 5,
    'pos_weight': 3.0,
    'buil_kernel1': 7
}
traincities=hyperparameters['train_cities']
output_dir = f'../../UNITAC-trained-models/multi_modal/{traincities}_DLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
buil_channels = hyperparameters['buil_channels']
buil_kernel = hyperparameters['buil_kernel1']
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_last=True,
    dirpath=output_dir,
    filename=f'multimodal_BCH{buil_channels}_BKR{buil_kernel}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=6,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=40)

model = MultiResolutionDeepLabV3(
    use_deeplnafrica=hyperparameters['use_deeplnafrica'],
    learning_rate=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay'],
    gamma=hyperparameters['gamma'],
    atrous_rates=hyperparameters['atrous_rates'],
    sched_step_size = hyperparameters['sched_step_size'],
    buil_channels = hyperparameters['buil_channels'],
    buil_kernel1 = hyperparameters['buil_kernel1'],
    pos_weight=torch.tensor(hyperparameters['pos_weight'], device='mps')
)
model.to(device)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=50,
    max_epochs=250,
    num_sanity_val_steps=3,
    precision='16-mixed',
    benchmark=True,
)
# Train the model
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation
# # best SD_DLV3/last.ckpt
best_model_path_dplv3 = os.path.join(grandparent_dir, "UNITAC-trained-models/multi_modal/all_DLV3/last-v4.ckpt")
# best_model_path_dplv3 = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3(buil_channels=buil_channels, buil_kernel1=buil_kernel)
checkpoint = torch.load(best_model_path_dplv3)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict, strict=True)
best_model = best_model.to(device)
best_model.eval()
check_nan_params(best_model)

# buildingsGeoDataset, _, _, _ = create_datasets(buildings_sceneSD, imgsize=512, stride = 256, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42, augment=False)
# sentinelGeoDataset, _, _, _ = create_datasets(sentinel_sceneSD, imgsize=256, stride = 128, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42,augment=False)
# sentinelGeoDataset_SD, buildingsGeoDataset_SD

predictions_iterator = MultiResPredictionsIterator(best_model, sent_val_ds_SD, buil_val_ds_SD, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=sentinel_sceneSD.extent,
    num_classes=len(class_config),
    smooth=True
)

gt_labels = sentinel_sceneSD.label_source.get_labels()

# Show predictions
fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels, threshold=0.5)
plt.show()

# save if needed
# fig.savefig('predictions_and_ground_truth.png', dpi=300, bbox_inches='tight')

# Evaluate against labels:
pred_labels_discrete = SemanticSegmentationDiscreteLabels.make_empty(
    extent=pred_labels.extent,
    num_classes=len(class_config))
scores = pred_labels.get_score_arr(pred_labels.extent)
pred_array_discrete = (scores > 0.5).astype(int)
pred_labels_discrete[pred_labels.extent] = pred_array_discrete[1]
evaluator = SemanticSegmentationEvaluator(class_config)
evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels_discrete)
inf_eval = evaluation.class_to_eval_item[1]
print(f"F1:{inf_eval.f1}")


# Function to calculate F1 score
def calculate_f1_score(model, sent_val_ds, buil_val_ds, device):
    predictions_iterator = MultiResPredictionsIterator(model, sent_val_ds, buil_val_ds, device=device)
    windows, predictions = zip(*predictions_iterator)

    # Create SemanticSegmentationLabels from predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
        extent=sent_val_ds.scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    gt_labels = sent_val_ds.scene.label_source.get_labels()

    # Evaluate against labels:
    pred_labels_discrete = SemanticSegmentationDiscreteLabels.make_empty(
        extent=pred_labels.extent,
        num_classes=len(class_config))
    scores = pred_labels.get_score_arr(pred_labels.extent)
    pred_array_discrete = (scores > 0.5).astype(int)
    pred_labels_discrete[pred_labels.extent] = pred_array_discrete[1]
    evaluator = SemanticSegmentationEvaluator(class_config)
    evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels_discrete)
    inf_eval = evaluation.class_to_eval_item[1]
    return inf_eval.f1

# List of cities and their corresponding val datasets
citie_valds = [
    ('Santo Domingo', sent_val_ds_SD, buil_val_ds_SD),
    ('Guatemala City', sent_val_ds_GC, buil_val_ds_GC),
    ('Tegucigalpa', sent_val_ds_TG, buil_val_ds_TG),
    ('Managua', sent_val_ds_MN, buil_val_ds_MN),
    ('Panama', sent_val_ds_PN, buil_val_ds_PN),
    ('San Salvador', sent_val_ds_SS, buil_val_ds_SS),
    ('San Jose', sent_val_ds_SJ, buil_val_ds_SJ),
    ('Belize City', sent_val_ds_BL, buil_val_ds_BL),
    ('Belmopan', sent_val_ds_BM, buil_val_ds_BM)
]

# Iterate through each city and calculate F1 score
for city_name, sent_val_ds, buil_val_ds in citie_valds:
    f1_score = calculate_f1_score(best_model, sent_val_ds, buil_val_ds, device)
    print(f"{city_name} - F1 Score: {f1_score:.4f}")


# # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# crs_transformer = RasterioCRSTransformer.from_uri(cities['TegucigalpaHND']['image_path'])
# gdf = gpd.read_file(cities['TegucigalpaHND']['labels_path'])
# gdf = gdf.to_crs('epsg:3857')
# xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
# affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
# crs_transformer.transform = affine_transform_buildings

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/multi-modal/all_DLV3/',
#     crs_transformer = crs_transformer,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)

# Visualise feature maps
# visualizer = FeatureMapVisualization(best_model)
# visualizer.add_hooks([
#     'buildings_encoder.0', # conv2d
#     'buildings_encoder.1', # batchnorm
#     'buildings_encoder.2', # relu
#     'buildings_encoder.3', # maxpool
#     'buildings_encoder.4', # maxpool
#     'encoder.conv1',
#     'encoder.layer1.0.conv1',
#     'encoder.layer4.2.conv3',
#     'segmentation_head.0',
#     'segmentation_head.1',
#     'segmentation_head.4',
#     'segmentation_head[-1]'
# ])

# # Get the iterator for the DataLoader
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
# data_iter = iter(val_loader)
# first_batch = next(data_iter)
# second_batch = next(data_iter)
# # third_batch = next(data_iter)

# x, y = vis_build.get_batch(val_buildings_dataset_SD, 2)
# vis_build.plot_batch(x, y, show=True)

# visualizer.visualize_feature_maps('buildings_encoder.0', second_batch, num_feature_maps=16) # conv2d
# visualizer.visualize_feature_maps('buildings_encoder.1', second_batch, num_feature_maps=16) # batchnorm
# visualizer.visualize_feature_maps('buildings_encoder.2', second_batch, num_feature_maps=16) # relu
# visualizer.visualize_feature_maps('buildings_encoder.3', second_batch, num_feature_maps=16) # maxpool
# visualizer.visualize_feature_maps('buildings_encoder.4', second_batch, num_feature_maps='all') #conv2d
# visualizer.visualize_feature_maps('encoder.layer1.0.conv1', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('encoder.layer4.2.conv3', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('segmentation_head.0', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('segmentation_head.1', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('segmentation_head.4', second_batch, num_feature_maps=16)
# visualizer.remove_hooks()


# # # Visualise filters
# # def visualize_filters(model, layer_name, num_filters=8):
#     # Get the layer by name
#     layer = dict(model.named_modules())[layer_name]
#     assert isinstance(layer, nn.Conv2d), "Layer should be of type nn.Conv2d"

#     # Get the weights of the filters
#     filters = layer.weight.data.clone().cpu().numpy()

#     # Normalize the filters to [0, 1] range for visualization
#     min_filter, max_filter = filters.min(), filters.max()
#     filters = (filters - min_filter) / (max_filter - min_filter)
    
#     # Plot the filters
#     num_filters = min(num_filters, filters.shape[0])  # Limit to number of available filters
#     fig, axes = plt.subplots(1, num_filters, figsize=(20, 10))
    
#     for i, ax in enumerate(axes):
#         filter_img = filters[i]
        
#         # If the filter has more than one channel, average the channels for visualization
#         if filter_img.shape[0] > 1:
#             filter_img = np.mean(filter_img, axis=0)
        
#         cax = ax.imshow(filter_img, cmap='viridis')
#         ax.axis('off')
    
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(cax, cax=cbar_ax)

#     plt.show()
    
# visualize_filters(best_model, 'segmentation_head.0', num_filters=8)

# # FLOPS
# # flops = FlopCountAnalysis(model, batch)

# # print(flops.total())
# # print(flops.by_module())

# # print(parameter_count_table(model))