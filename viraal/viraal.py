import hydra

@hydra.main(config_path='config/config.yaml')
def viraal(cfg):
    print(cfg.pretty())

if __name__ == "__main__":
    viraal()