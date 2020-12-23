% Given the file name, reads data from .txt files
function [data] = load_data(filename)
data = readtable(filename);
data = table2array(data);