



def create_training_selector():
    return "training selector"




def create_testing_selector():
    return "testing selector"

class testing_selector:
    """Kuiper's testing selector

       We provide two kinds of selector:
       TODO: add description
    """
    def __init__(self):
        pass
    
    def select_by_deviation(self, dev_target , range_of_capacity , total_num_clients):
        return 0

    def update_client_info(self, client_id , client_info):
        return 0
    
    def select_by_category(self, request_list , testing_config):
        return 0