import os
import sys
import shutil
import sysconfig
if sys.platform == 'win32':
    from win32com.client import Dispatch
    import winreg

import sesame


DESKTOP_FOLDER = get_special_folder_path("CSIDL_DESKTOPDIRECTORY")
NAME = 'Sesame.lnk'

if sys.argv[1] == 'install':
    create_shortcut(
        os.path.join(sys.prefix, 'pythonw.exe'), # program
        'Description of the shortcut', # description
        NAME, # filename
        sesame.__path__[0] + '/sesame/app.py', # parameters
        '', # workdir
        # os.path.join(os.path.dirname(sesame.__file__), 'favicon.ico'), # iconpath
    )
    # move shortcut from current directory to DESKTOP_FOLDER
    shutil.move(os.path.join(os.getcwd(), NAME),
                os.path.join(DESKTOP_FOLDER, NAME))
    # tell windows installer that we created another
    # file which should be deleted on uninstallation
    file_created(os.path.join(DESKTOP_FOLDER, NAME))

if sys.argv[1] == 'remove':
    pass
    # This will be run on uninstallation. Nothing to do.
