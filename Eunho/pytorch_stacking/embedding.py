from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_tabular.models.common.heads import LinearHeadConfig

class TabularPipeline:
    def __init__(self, train_fold, val_fold, test_fold, seed, numeric_cols, cat_cols):
        # 데이터셋 및 설정 변수 초기화
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.seed = seed
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols

        # DataConfig 설정
        self.data_config = DataConfig(
            target=["임신 성공 확률"],
            continuous_cols=self.numeric_cols,
            categorical_cols=self.cat_cols,
            # continuous_feature_transform="quantile_normal",  # 필요 시 주석 해제
            normalize_continuous_features=False,
        )
        
        # TrainerConfig 설정
        self.trainer_config = TrainerConfig(
            auto_lr_find=True,  
            batch_size=4096,
            max_epochs=50,
            early_stopping="valid_loss",     
            early_stopping_mode="min",
            early_stopping_patience=3,
            checkpoints="valid_loss",        
            load_best=True, 
            devices=-1,  # -1은 모든 사용 가능한 장치 사용
            seed=self.seed
        )

        # OptimizerConfig 설정
        self.optimizer_config = OptimizerConfig()

        # LinearHeadConfig를 dict로 변환
        head_config = LinearHeadConfig(
            layers="",  # 헤드에 추가 레이어 없이 출력 차원으로 매핑
            dropout=0.1,
            initialization="kaiming",
        ).__dict__

        # 모델 구성 설정
        self.model_config = CategoryEmbeddingModelConfig(
            task="regression",
            layers="512-256-16",  # 각 레이어의 노드 수
            activation="LeakyReLU",  # 레이어 간 활성화 함수
            dropout=0.1,
            initialization="kaiming",
            head="LinearHead",  # Linear Head 사용
            head_config=head_config,  # Linear Head의 설정
            learning_rate=1e-3,
            # metrics=["accuracy", "f1_score", "auc"],  # 필요 시 주석 해제
            # metrics_params=[{}, {}, {}],
            # metrics_prob_input=[False, True, True],
        )

        # TabularModel 초기화
        self.tabular_model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
            verbose=False,
        )

        # 이후에 사용할 변수들 초기화
        self.datamodule = None
        self.model = None
        self.transformer = None
        self.fold_train_trans = None
        self.fold_valid_trans = None
        self.fold_test_trans = None

    def prepare_data(self):
        """데이터로더와 모델을 준비합니다."""
        self.datamodule = self.tabular_model.prepare_dataloader(
            train=self.train_fold,
            validation=self.val_fold,
            seed=self.seed
        )
        self.model = self.tabular_model.prepare_model(self.datamodule)

    def train_model(self):
        """모델 학습을 수행합니다."""
        if self.datamodule is None or self.model is None:
            self.prepare_data()
        self.tabular_model.train(self.model, self.datamodule)

    def transform_data(self):
        """학습된 모델을 이용해 범주형 임베딩을 추출합니다."""
        self.transformer = CategoricalEmbeddingTransformer(self.tabular_model)
        self.fold_train_trans = self.transformer.fit_transform(self.train_fold)
        self.fold_valid_trans = self.transformer.transform(self.val_fold)
        self.fold_test_trans = self.transformer.transform(self.test_fold)
        return self.fold_train_trans, self.fold_valid_trans, self.fold_test_trans

    def run_pipeline(self):
        """전체 파이프라인(데이터 준비, 학습, 임베딩 추출)을 실행합니다."""
        self.prepare_data()
        self.train_model()
        return self.transform_data()
