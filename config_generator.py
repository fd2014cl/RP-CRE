import configparser

if __name__ == '__main__':
    # This file is for generating model configs.
    # Please run this file before running main.py.

    # initialize config
    config = configparser.ConfigParser()

    # add parameter settings for the pretraining model
    config['training'] = {}
    config['training']['batch_size'] = '64'
    config['training']['gradient_accumulation_steps'] = '4'
    config['training']['total_round'] = '5'
    # config['training']['rel_per_task'] = '8'
    config['training']['rel_per_task'] = '4'    
    config['training']['drop_out'] = '0.5'
    config['training']['num_workers'] = '2'
    config['training']['step1_epochs'] = '10'
    config['training']['step2_epochs'] = '10'
    config['training']['step3_epochs'] = '10'
    config['training']['num_protos'] = '10'
    config['training']['device'] = 'cuda'
    config['training']['seed'] = '2021'
    config['training']['max_grad_norm'] = '10'

    config['Encoder'] = {}
    config['Encoder']['bert_path'] = './pretrained_models/bert-base-uncased'
    config['Encoder']['max_length'] = '256'
    config['Encoder']['vocab_size'] = '30522'
    config['Encoder']['marker_size'] = '4'
    config["Encoder"]['pattern'] = 'entity_marker'
    config["Encoder"]['encoder_output_size'] = '768'

    config['memory'] = {}
    config['memory']['key_size'] = '256'
    config['memory']['head_size'] = '768'
    config['memory']['mem_size'] = '768'

    config['data'] = {}
    config['data']['task_name'] = 'TACRED'
    # config['data']['data_file'] = './data/data_with_marker.json'
    # config['data']['relation_file'] = './data/id2rel.json'
    # config['data']['num_of_relation'] = '80'
    config['data']['data_file'] = './data/data_with_marker_tacred.json'
    config['data']['relation_file'] = './data/id2rel_tacred.json'
    config['data']['num_of_relation'] = '40'   
    config['data']['num_of_train'] = '420'
    config['data']['num_of_val'] = '140'
    config['data']['num_of_test'] = '140'
    
    # save config files
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)