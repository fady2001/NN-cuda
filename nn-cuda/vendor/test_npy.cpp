#include "npy.hpp"
#include <vector>
#include <string>

int main()
{

    const std::string path{"data.npy"};
    npy::npy_data<float> d = npy::read_npy<float>(path);

    std::vector<float> data = d.data;
    std::vector<unsigned long> shape = d.shape;
    bool fortran_order = d.fortran_order;
    // print data
    for (auto i : data)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // print shape
    for (auto i : shape)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // print fortran_order
    std::cout << fortran_order << std::endl;

}
