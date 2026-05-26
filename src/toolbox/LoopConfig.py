import numpy as np 

class LoopConfig:

    LOOP_DEFAULT = {
        "N_annotated": 500,
        "sampling_method": {"balance": "random", "stratified": None},
        "splits_ratio": [80, 10, 10],

        "model_name": "google-bert/bert-base-uncased",
        "n_epochs": 4, 
        "learning_rate": 3e-5,
        "weight_decay": 0,
        "batch_size": 8,

        "output_dir": "./models/current",
        "seed": 42,
        "device_batch_size": 4,
        "test_mode": False,
    }

    VARIABLES_TYPE = {
        "N_annotated": int,
        "sampling_method": dict, # Specific case handled in self.__extract_value
        "splits_ratio": list[int], # Specific case handled in self.__extract_value

        "model_name": str, 
        "n_epochs": int,
        "learning_rate": float, 
        "weight_decay": float,
        "batch_size": int, 

        "output_dir": str, 
        "seed": int,
        "device_batch_size": int, 
        "test_mode": bool
    }

    VARIABLES_TO_CHECK_FOR_EQUALITY = [
        "task_name", 
        "dichotomization_label", 

        "N_annotated", 
        "sampling_method", 
        "splits_ratio", 
        
        "model_name", 
        "n_epochs",
        "learning_rate", 
        "weight_decay", 
        "batch_size"
    ]

    def __extract_value(self, param_name:str, **kwargs):
        if param_name == "splits_ratio":
            splits_ratio_as_list = list(kwargs.get("splits_ratio", self.LOOP_DEFAULT["splits_ratio"]))
            try: 
                out = [int(v) for v in splits_ratio_as_list]
                return out 
            except: 
                raise ValueError((f"Error parsing splits_ratio, should receive a "
                    f"list of ints, received {kwargs.get('splits_ratio', None)}"))
        if param_name == "sampling_method":
            sampling_method = kwargs.get("sampling_method", self.LOOP_DEFAULT["splits_ratio"])
            try:
                sampling_method = dict(sampling_method)
                return{
                    "balance": sampling_method.get("balance", self.LOOP_DEFAULT["sampling_method"]["balance"]),
                    "stratified": sampling_method.get("stratified", self.LOOP_DEFAULT["sampling_method"]["stratified"])
                }
            except:
                raise ValueError((f"Error parsing sampling method format, "
                    f"should be a dictionary but received: {kwargs.get('sampling_method', None)}"))
        try: 
            out = self.VARIABLES_TYPE[param_name](
                kwargs.get(param_name, self.LOOP_DEFAULT[param_name])
            )
            return out 
        except:
            raise ValueError((f"Error parsing {param_name}, should be a "
                f"{self.VARIABLES_TYPE[param_name]} but received {kwargs.get(param_name, None)}"))


    def __init__(self, task_name : str, dichotomization_label : str, **kwargs) -> None:
        """
        Takes in any kwargs and return a dictionnary with the expected keys, default 
        values and format
        """
        self.task_name = str(task_name)
        self.dichotomization_label = str(dichotomization_label)

        self.N_annotated = self.__extract_value("N_annotated", **kwargs)
        self.sampling_method = self.__extract_value("sampling_method", **kwargs)
        self.splits_ratio = self.__extract_value("splits_ratio", **kwargs)

        self.model_name = self.__extract_value("model_name", **kwargs)
        self.n_epochs = self.__extract_value("n_epochs", **kwargs)
        self.learning_rate = self.__extract_value("learning_rate", **kwargs)
        self.weight_decay = self.__extract_value("weight_decay", **kwargs)
        self.batch_size = self.__extract_value("batch_size", **kwargs)
        
        self.output_dir = self.__extract_value("output_dir", **kwargs)
        self.seed = self.__extract_value("seed", **kwargs)
        self.device_batch_size = self.__extract_value("device_batch_size", **kwargs)
        self.test_mode = self.__extract_value("test_mode", **kwargs)

    def to_dict(self) -> dict:
        return {key : self.__getattribute__(key) for key in self.VARIABLES_TO_CHECK_FOR_EQUALITY}
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, LoopConfig):
            return TypeError("Can only check equality with LOOP_CONFIG objects")
        check_list = [
            self.__getattribute__(key) == __value.__getattribute__(key)
            for key in self.VARIABLES_TO_CHECK_FOR_EQUALITY
        ]
        return np.array(check_list).all()

    def __str__(self) -> bool: 
        return " | ".join([f'{k}:{v}' for k,v in self.to_dict().items()])