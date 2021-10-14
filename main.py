from CL_prediction import seq_cl
from process_data import read_data_covid

if __name__ == "__main__":
    """
    data: Input data, must be nxm, where n is time dimension, m is feature dimension
    label: the corresponding label associated with data
    """
    read_d = read_data_covid(data,label)
    seq = seq_cl(read_d)
    seq.config_model()
    seq.train()