from .utils.lp import *



def create_training_selector():
    return "training selector"




def create_testing_selector():
    return "testing selector"

class testing_selector:
    """Kuiper's testing selector

    We provide two kinds of selector:
    select_by_deviation: 
    select_by_category: 

    Attributes:
        client_info: Optional; A dictionary that stores client id to client profile(system speech and
            network bandwidth) mapping. For example, {1: [153.0, 2209.61]} indicates that client 1
            needs 153ms to run a single sample inference and their network bandwidth is 2209 Kbps.
        model_size: Optional; the size of the model(i.e., the data transfer size) in kb
        data_distribution: Optional; individual data characteristics(distribution). 

    """
    def __init__(self, data_distribution=None, client_info=None, model_size=None):
        """Inits testing selector."""
        self.client_info = client_info
        self.model_size = model_size
        self.data_distribution = data_distribution
    

    def update_client_info(self, client_ids, client_profile):
        """Update clients' profile(system speed and network bandwidth)

        Since the clients' info is dynamic, developers can use this function
        to update clients' profile. If the client id does not exist, Kuiper will 
        create a new entry for this client.

        Args:
            client_ids: A list of client ids whose profile needs to be updated
            client_info: Updated information about client profile, formatted as 
                a list of pairs(speed, bw)
        
        Raises:
            Raises an error if len(client_ids) != len(client_info)
        """
        return 0
    
    def select_by_deviation(self, dev_target, range_of_capacity, max_num_clients):
        """Testing selector that preserves data representativeness.

        Given the developer-specified tolerance `dev_target`, Kuiper can estimate the number 
        of participants needed such that the deviation from the representative categorical 
        distribution is bounded.

        Args:
            dev_target: developer-specified tolerance
            range_of_capacity: the global max-min range of number of samples across all clients
            TODO: 

        Returns:
            The estimated number of participant needed to satisfy developer's requirement
        """
        selected_client_ids = []
        return selected_client_ids
    
    def select_by_category(self, request_list, max_num_clients=None):
        """Testing selection based on requested number of samples per category.

        When individual data characteristics(distribution) is provided, Kuiper can 
        enforce client's request on the number of samples per category. 

        Args:
            request_list: a list that specifies the desired number of samples per category. 
                i.e., [num_requested_samples_class_x for class_x in request_list]. 
            max_num_clients: Optional; the maximum number of participants .

        Returns:
            A list of selected participants ids.

        Raises:
            Raises an error if 1) no client information is provided or 2) the requirement 
            cannot be satisfied(e.g., max_num_clients too small).
        """
        # TODO: Add error handling
        selected_client_ids = run_select_by_category(request_list, self.data_distribution, 
            self.client_info, max_num_clients, self.model_size)
        return selected_client_ids