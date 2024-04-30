# pip install pytorch-fid

python -m pytorch_fid fid/real/ fid/fake/vae_0 # FID: 106.26333420461253
python -m pytorch_fid fid/real/ fid/fake/vae_0.1 # FID: 108.42034713456962
python -m pytorch_fid fid/real/ fid/fake/vae_0.3 # FID: 108.3695723237393
python -m pytorch_fid fid/real/ fid/fake/vae_0.5 # FID: 117.58364390978278

python -m pytorch_fid fid/real/ fid/fake/gan_0 # FID: 123.54347509355114
python -m pytorch_fid fid/real/ fid/fake/gan_0.1 # FID: 154.9873005559262
python -m pytorch_fid fid/real/ fid/fake/gan_0.3 # FID: 259.5945890145011
python -m pytorch_fid fid/real/ fid/fake/gan_0.5 # FID: 299.3219627565743

python -m pytorch_fid fid/real/ fid/fake/dm_0 # FID: 46.01468036499398
python -m pytorch_fid fid/real/ fid/fake/dm_0.1 # FID: 153.03076453221735
python -m pytorch_fid fid/real/ fid/fake/dm_0.3 # FID: 295.1775928185118
python -m pytorch_fid fid/real/ fid/fake/dm_0.5 # FID: 392.6717186974191