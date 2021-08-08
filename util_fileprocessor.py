
def write_list_of_lists(filename, Y):
    """
    function  data into a file
    @param filename: the name of the file to be created
    @param Y: the list of lists with data
    """
    f = open(filename, "w")
    for data in Y:
        temp_string = ""
        for i in range(len(data) - 1):
            temp_string += str(data[i]) + " "
        temp_string += str(data[-1]) + "\n"
        f.write(temp_string)
    f.close()
    return


def append_list_of_lists(filename, Y):
    """
    function  data into a file
    @param filename: the name of the file to be created
    @param Y: the list of lists with data
    """
    f = open(filename, "a")
    for data in Y:
        temp_string = ""
        for i in range(len(data) - 1):
            temp_string += str(data[i]) + " "
        temp_string += str(data[-1]) + "\n"
        f.write(temp_string)
    f.close()
    return


def read_from_file(filename):
    """
    function to read all data from a file
    @param filename: the name of the file
    @return : the data as a list
    """
    Y = []
    f = open(filename, "r")
    for line in f:
        data = line.split()
        data_ = []
        for x in data:
            data_.append(int(x))
        Y.append(data_)
    f.close()
    return Y


def read_list_from_file(filename):
    Y = []
    f = open(filename, 'r')
    for line in f:
        data = line.split()
        data = int(data[0])
        Y.append(data)
    f.close()
    return Y


def print_each_line(filename):
    """ 
    function to print out all data read froma file
    @param filename: the name of the file to be read
    """
    data = read_from_file(filename)
    for element in data:
        print(element)
    return


def write_list(filename, Y):
    f = open(filename, "w")
    temp_string = ""
    for data in Y:
        temp_string += str(data) + "\n"
    f.write(temp_string)
    f.close()
    return

# --------- dictionaries ---------------------------

"""
#python 2.7 version
def writeDictToFile(filename,Y):
    f = open(filename,"w")
    temp_string = ""
    for key, value in sorted(Y.iteritems(), key=lambda (k,v): (v,k)):
        temp_string += str(key)+" : "+str(value)+ "\n"
    f.write(temp_string)
    f.close()
    return
"""
# python 3.X version


def write_dict(filename, Y):
    f = open(filename, "w")
    temp_string = ""
    for key in Y:
        value = Y[key]
        temp_string += str(key) + " : " + str(value) + "\n"
    f.write(temp_string)
    f.close()
    return


def read_dict(filename):
    _dict = dict()  # create an empty dictionary
    f = open(filename, "r")
    for line in f:
        data = line.split(" : ")
        _index = data[1].split('\n')[0]
        _dict[str(data[0])] = str(int(_index))
    return _dict


def write_dict_with_list(filename, Y):
    f = open(filename, "w")
    for key in Y:
        temp_string = ""
        temp_string = str(key)
        value = Y[key]
        for data in value:
            temp_string += " " + str(data)
        temp_string += "\n"
        f.write(temp_string)
    f.close()
    return


def read_dict_with_list(filename):
    f = open(filename, "r")
    dict_ = dict()
    for line in f:
        data = line.split()
        key_ = str(data[0])
        values_ = []
        for i in data:
            values_.append(int(i))
        dict_[key_] = values_
    return dict_
