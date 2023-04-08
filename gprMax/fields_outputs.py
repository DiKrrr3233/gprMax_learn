# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

from string import Template

import h5py

from gprMax._version import __version__
from gprMax.grid import Ix, Iy, Iz


def store_outputs(iteration, Ex, Ey, Ez, Hx, Hy, Hz, G):
    """Stores field component values for every receiver and transmission line.
    
    您选择的文本是一个名为 store_outputs 的函数，它用于存储每个接收器和传输线的场分量值。
    该函数接受当前迭代次数、电场和磁场值以及一个包含模型描述的必要参数的网格类实例作为输入。
    该函数遍历所有接收器并存储电场或磁场分量，如果输出中包含 ‘I’，则存储电流分量。
    对于传输线，该函数存储总电压和总电流
    
    Args:
        iteration (int): Current iteration number.
        Ex, Ey, Ez, Hx, Hy, Hz (memory view): Current electric and magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    for rx in G.rxs:
        for output in rx.outputs:
            # Store electric or magnetic field components
            if 'I' not in output:
                field = locals()[output]
                rx.outputs[output][iteration] = field[rx.xcoord, rx.ycoord, rx.zcoord]
            # Store current component
            else:
                func = globals()[output]
                rx.outputs[output][iteration] = func(rx.xcoord, rx.ycoord, rx.zcoord, Hx, Hy, Hz, G)

    for tl in G.transmissionlines:
        tl.Vtotal[iteration] = tl.voltage[tl.antpos]
        tl.Itotal[iteration] = tl.current[tl.antpos]


kernel_template_store_outputs = Template("""

kernel_template_store_outputs 的模板，它定义了一个 CUDA 内核，用于在 GPU 上执行 store_outputs 函数的功能。
这个内核使用宏来将下标转换为线性索引，并定义了一个名为 store_outputs 的全局函数，
该函数接受接收器总数、迭代次数、接收器坐标、接收器数组以及电场和磁场数组作为输入，并存储每个接收器的场分量值。

// Macros for converting subscripts to linear index:
#define INDEX2D_RXCOORDS(m, n) (m)*($NY_RXCOORDS)+(n)
#define INDEX3D_RXS(i, j, k) (i)*($NY_RXS)*($NZ_RXS)+(j)*($NZ_RXS)+(k)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)

//////////////////////////////////////////////////////
// Stores field component values for every receiver //
//////////////////////////////////////////////////////

__global__ void store_outputs(int NRX, int iteration, const int* __restrict__ rxcoords, $REAL *rxs, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz) {

    //  This function stores field component values for every receiver in the model.
    //
    //  Args:
    //      NRX: Total number of receivers in the model
    //      rxs: Array to store field components for receivers - rows are field components; columns are iterations; pages are receivers
    //      E, H: Access to field component arrays

    // Obtain the linear index corresponding to the current thread and use for each receiver
    int rx = blockIdx.x * blockDim.x + threadIdx.x;

    int i, j, k;

    if (rx < NRX) {
        i = rxcoords[INDEX2D_RXCOORDS(rx,0)];
        j = rxcoords[INDEX2D_RXCOORDS(rx,1)];
        k = rxcoords[INDEX2D_RXCOORDS(rx,2)];
        rxs[INDEX3D_RXS(0,iteration,rx)] = Ex[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(1,iteration,rx)] = Ey[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(2,iteration,rx)] = Ez[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(3,iteration,rx)] = Hx[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(4,iteration,rx)] = Hy[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(5,iteration,rx)] = Hz[INDEX3D_FIELDS(i,j,k)];
    }
}

""")


def write_hdf5_outputfile(outputfile, G):
    """Write an output file in HDF5 format.
    write_hdf5_outputfile 的函数，它用于将模拟结果写入 HDF5 格式的输出文件。
    
    该函数接受输出文件的名称和一个包含模型描述的必要参数的网格类实例作为输入。
    
    该函数首先创建一个 HDF5 文件并添加一些属性，如 gprMax 版本、模型标题、迭代次数、网格大小、空间分辨率、时间步长、源数量、接收器数量、源步长和接收器步长等。
    然后，该函数为除传输线外的所有源创建组，并添加类型和位置数据属性。对于传输线，该函数创建组并添加位置数据、线电阻和线离散化属性，并写入线电压和电流数组。
    最后，该函数为接收器创建组，添加位置数据并写入场分量数组。
    
    Args:
        outputfile (str): Name of the output file.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    f = h5py.File(outputfile, 'w')
    f.attrs['gprMax'] = __version__
    f.attrs['Title'] = G.title
    f.attrs['Iterations'] = G.iterations
    f.attrs['nx_ny_nz'] = (G.nx, G.ny, G.nz)
    f.attrs['dx_dy_dz'] = (G.dx, G.dy, G.dz)
    f.attrs['dt'] = G.dt
    nsrc = len(G.voltagesources + G.hertziandipoles + G.magneticdipoles + G.transmissionlines)
    f.attrs['nsrc'] = nsrc
    f.attrs['nrx'] = len(G.rxs)
    f.attrs['srcsteps'] = G.srcsteps
    f.attrs['rxsteps'] = G.rxsteps

    # Create group for sources (except transmission lines); add type and positional data attributes
    srclist = G.voltagesources + G.hertziandipoles + G.magneticdipoles
    for srcindex, src in enumerate(srclist):
        grp = f.create_group('/srcs/src' + str(srcindex + 1))
        grp.attrs['Type'] = type(src).__name__
        grp.attrs['Position'] = (src.xcoord * G.dx, src.ycoord * G.dy, src.zcoord * G.dz)

    # Create group for transmission lines; add positional data, line resistance and
    # line discretisation attributes; write arrays for line voltages and currents
    for tlindex, tl in enumerate(G.transmissionlines):
        grp = f.create_group('/tls/tl' + str(tlindex + 1))
        grp.attrs['Position'] = (tl.xcoord * G.dx, tl.ycoord * G.dy, tl.zcoord * G.dz)
        grp.attrs['Resistance'] = tl.resistance
        grp.attrs['dl'] = tl.dl
        # Save incident voltage and current
        grp['Vinc'] = tl.Vinc
        grp['Iinc'] = tl.Iinc
        # Save total voltage and current
        f['/tls/tl' + str(tlindex + 1) + '/Vtotal'] = tl.Vtotal
        f['/tls/tl' + str(tlindex + 1) + '/Itotal'] = tl.Itotal

    # Create group, add positional data and write field component arrays for receivers
    for rxindex, rx in enumerate(G.rxs):
        grp = f.create_group('/rxs/rx' + str(rxindex + 1))
        if rx.ID:
            grp.attrs['Name'] = rx.ID
        grp.attrs['Position'] = (rx.xcoord * G.dx, rx.ycoord * G.dy, rx.zcoord * G.dz)

        for output in rx.outputs:
            f['/rxs/rx' + str(rxindex + 1) + '/' + output] = rx.outputs[output]
