t_mode='gist_adapter' # linear_probe FT gist_adapter
model='vit_base_patch16_224_in21k'
CUDA_NUM='0,1'
NODE_NUM=2
DATA_ROOT_PATH=/path_to/vtab-1k






CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14337  \
	train.py ${DATA_ROOT_PATH}/caltech101  \
    --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/caltech101/${t_mode} \
	--amp  --tuning-mode $t_mode --pretrained  \

CUDA_VISIBLE_DEVICES=$CUDA_NUM, python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=19547  \
	train.py ${DATA_ROOT_PATH}/cifar  \
    --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/cifar_100/${t_mode} \
	--amp  --tuning-mode $t_mode --pretrained  \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14332  \
	train.py ${DATA_ROOT_PATH}/clevr_count \
    --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/clevr_count/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=10032  \
	train.py ${DATA_ROOT_PATH}/clevr_dist  \
    --dataset clevr_dist --num-classes 6  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/clevr_dist/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=26662  \
	train.py ${DATA_ROOT_PATH}/diabetic_retinopathy  \
    --dataset diabetic_retinopathy --num-classes 5  --no-aug --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/diabetic_retinopathy/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM, python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=13002  \
	train.py ${DATA_ROOT_PATH}/dmlab  \
    --dataset dmlab --num-classes 6  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/dmlab/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=12102  \
	train.py ${DATA_ROOT_PATH}/dsprites_loc  \
    --dataset dsprites_loc --num-classes 16  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/dsprites_loc/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=12002  \
	train.py ${DATA_ROOT_PATH}/dsprites_ori  \
    --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/dsprites_ori/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14312  \
	train.py ${DATA_ROOT_PATH}/dtd  \
    --dataset dtd --num-classes 47  --no-aug --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
    --output  output/${model}/vtab/dtd/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14112  \
	train.py ${DATA_ROOT_PATH}/eurosat  \
    --dataset eurosat --num-classes 10  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 3e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/eurosat/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14222  \
	train.py ${DATA_ROOT_PATH}/oxford_flowers102 \
    --dataset flowers102 --num-classes 102  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/flowers102/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM --master_port=14332  \
	train.py ${DATA_ROOT_PATH}/kitti  \
    --dataset kitti --num-classes 4  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/kitti/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  \

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14332  \
	train.py ${DATA_ROOT_PATH}/patch_camelyon  \
    --dataset patch_camelyon --num-classes 2  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/patch_camelyon/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14332  \
	train.py ${DATA_ROOT_PATH}/oxford_iiit_pet  \
    --dataset pets --num-classes 37  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/pets/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained 

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=11222  \
	train.py ${DATA_ROOT_PATH}/resisc45  \
    --dataset resisc45 --num-classes 45  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/resisc45/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained 

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14882  \
	train.py ${DATA_ROOT_PATH}/smallnorb_azi  \
    --dataset smallnorb_azi --num-classes 18  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/smallnorb_azi/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=24332  \
	train.py ${DATA_ROOT_PATH}/smallnorb_ele  \
    --dataset smallnorb_ele --num-classes 9  --no-aug  --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/smallnorb_ele/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained 

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14192  \
	train.py ${DATA_ROOT_PATH}/sun397  \
    --dataset sun397 --num-classes 397  --no-aug --direct-resize  --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/sun397/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  

CUDA_VISIBLE_DEVICES=$CUDA_NUM,  python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=14332  \
	train.py ${DATA_ROOT_PATH}/svhn  \
    --dataset svhn --num-classes 10  --no-aug --direct-resize --model $model  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/${model}/vtab/svhn/${t_mode} \
	--amp --tuning-mode $t_mode --pretrained  