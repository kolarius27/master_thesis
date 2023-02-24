from opals import Import, Cell, View, Histo, Types
import os
import sys


def main():
    # files
    print('creating paths')
    path = r'E:\NATUR_CUNI\_DP\data\LAZ'
    input_file = os.path.join(path, r'ahk_uls_full.las')
    odm_file = os.path.join(path, r'odm\ahk_uls_full.odm')
    pdens_file = os.path.join(path, r'raster\ahk_uls_pdens.tif')
    hist_file = os.path.join(path, r'info\ahk_uls_full_hist.svg')
    hist_pdens_file = os.path.join(path, r'info\ahk_uls_pdens_hist.svg')
    print('paths created')

    # loading opals objects
    print('loading opals objects')
    imp = Import.Import()
    cell = Cell.Cell()
    view = View.View()
    hist = Histo.Histo()

    # enabling log warnings
    imp.commons.screenLogLevel = Types.LogLevel.warning
    cell.commons.screenLogLevel = Types.LogLevel.warning
    view.commons.screenLogLevel = Types.LogLevel.warning
    hist.commons.screenLogLevel = Types.LogLevel.warning
    print('opals objects loaded')

    # odm
    print('creating ODM')
    imp.inFile = input_file
    imp.outFile = odm_file
    imp.run()
    print('ODM created')

    # pdens
    print('creating pdens')
    cell.inFile = odm_file
    cell.outFile = pdens_file
    cell.cellSize = 1.0
    cell.feature = 'pdens'
    cell.filter = 'Echo[last]'
    cell.run()
    print('pdens created')

    # histogram
    print('histogram of odm')
    hist.inFile = odm_file
    hist.plotFile = hist_file
    hist.run()
    print('histogram created')

    print('histogram of pdens')
    hist.inFile = pdens_file
    hist.plotFile = hist_pdens_file
    hist.run()
    print('pdens created')

if __name__ == "__main__":
    main()
