def points_to_ply(data, file):
    f = open(file, 'w')

    f.write('ply\n' + 'format ascii 1.0\n')
    f.write('element vertex ' + str(data.shape[0]) + '\n')
    f.write('property float x\n' + 'property float y\n' + 'property float z\n' + 'end_header\n')

    for i in range(data.shape[0]):
            f.write(str(data[i, 0]) + ' ')
            f.write(str(data[i, 1]) + ' ')
            f.write(str(data[i, 2]) + '\n')
