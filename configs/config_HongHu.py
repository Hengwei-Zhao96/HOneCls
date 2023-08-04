config = dict(
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=270,
            num_classes=1,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewOCCHongHuLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./UAVData/data',  # Please modify the path of the data
                gt_mat_path='./UAVData/gt',  # Please modify the path of the data
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=1,
                cls=None,
                ratio=40
            )
        ),
        test=dict(
            type='NewOCCHonghuLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./UAVData/data',  # Please modify the path of the data
                gt_mat_path='./UAVData/gt',  # Please modify the path of the data
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=1,
                cls=None,
                ratio=40
            )
        )
    ),
    risk_estimation=dict(
        type=None,
        class_prior=None,
        class_weight=None,
        warm_up_epoch=20,
        focal_weight=None,
        loss='sigmoid'
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            momentum=0.9,
            weight_decay=0.0001
        )
    ),
    learning_rate=dict(
        type='ExponentialLR',
        params=dict(
            base_lr=0.001,
            power=1,
            max_iters=1000),
    ),
    out_config=dict(
        params=dict(
            image_size=(678, 465),
            palette=[
                [0, 0, 0],
                [255, 0, 0],
                [255, 255, 255],
                [176, 48, 96],
                [255, 255, 0],
                [255, 127, 80],
                [0, 255, 0],
                [0, 205, 0],
                [0, 139, 0],
                [127, 255, 212],
                [160, 32, 240],
                [216, 191, 216],
                [0, 0, 255],
                [0, 0, 139],
                [218, 112, 214],
                [160, 82, 45],
                [0, 255, 255],
                [255, 165, 0],
                [127, 255, 0],
                [139, 139, 0],
                [0, 139, 139],
                [205, 181, 205],
                [238, 154, 0]],
        ),
    )
)
