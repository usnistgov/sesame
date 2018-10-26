# -*- mode: python -*-

block_cipher = None


a = Analysis(['app.py'],
             pathex=['C:\\Users\\phaney\\PycharmProjects\\sesame_3'],
             binaries=[],
             datas=[('C:\\Users\\phaney\\PycharmProjects\\sesame_3\\sesame\\ui\\resources\\logo-icon_sesame.png', 'resources')],
             hiddenimports=['sesame.getFandJ_eq', 'ipykernel.datapub', 'sesame.getF', 'sesame.jacobian1', 'sesame.jacobian','sesame.builder', 'sesame.analyzer', 'PyQt5.sip'
			],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='sesame',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False,
	  icon='C:\\Users\\phaney\\PycharmProjects\\sesame_3\\sesame\\ui\\resources\\logo-icon_sesame.ico')
