"""
Generate channels locations in 3D cartesian coordinates
similar to the provided 2D cartesian coordinates.
"""

import mne
from matplotlib import pyplot as plt

from dataset import load_dataset, PATIENTS

dataset = load_dataset(PATIENTS[0])

# show provided sensor locations in 2d cartesian coordinates
fig = dataset.get_mne_layout().plot(show=False)
plt.savefig('original2d.pdf')
plt.close(fig)

# obtain 3d cartesian coordinates from standard 1005 EEG layout
standard_1005 = mne.channels.read_montage('standard_1005')

fig = standard_1005.plot(show=False, scale_factor=0)
plt.savefig('standard_1005.pdf')
plt.close(fig)

# rename some channels for `standard_1005` layout
# old name (provided name) -> new name (name from `standard_1005` layout)
change_names = {
    'FAF5': 'AFF5h',
    'FAF1': 'AFF1h',
    'FAF2': 'AFF2h',
    'FAF6': 'AFF6h',

    'FFC7': 'FFT7h',
    'FFC8': 'FFT8h',

    'CFC7': 'FTT7',
    'CFC8': 'FTT8',

    'CFC5': 'FCC5h',
    'CFC3': 'FCC3h',
    'CFC1': 'FCC1h',
    'CFC2': 'FCC2h',
    'CFC4': 'FCC4h',
    'CFC6': 'FCC6h',

    'CCP7': 'TTP7',
    'CCP8': 'TTP8',

    'PCP7': 'TPP7h',
    'PCP8': 'TPP8h',

    'PCP5': 'CPP5h',
    'PCP3': 'CPP3h',
    'PCP1': 'CPP1h',
    'PCP2': 'CPP2h',
    'PCP4': 'CPP4h',
    'PCP6': 'CPP6h',

    'OPO1': 'POO1',
    'OPO2': 'POO2',
}
channel_names = dataset.channel_names.copy()
for old, new in change_names.items():
    channel_names[channel_names.index(old)] = new

# check that we have all the correct channel names
for ch in channel_names:
    if ch not in standard_1005.ch_names:
        assert True
assert len(set(channel_names)) == dataset.n_channels

# save new coordinates in EEGLAB format .xyz
with open('locations.xyz', 'w') as f:
    for idx, (new_name, old_name) in enumerate(zip(channel_names, dataset.channel_names)):
        montage_idx = standard_1005.ch_names.index(new_name)

        f.write('{} {} {} {} {}\n'.format(
            idx,
            standard_1005.pos[montage_idx][0],
            standard_1005.pos[montage_idx][1],
            standard_1005.pos[montage_idx][2],
            old_name
        ))

# plot montage of 3d cartesian coordinates
montage = mne.channels.Montage(
    pos=standard_1005.pos[[standard_1005.ch_names.index(ch) for ch in channel_names]],
    ch_names=dataset.channel_names,  # use original names
    kind='eeg', selection=list(range(dataset.n_channels)),
    nasion=standard_1005.nasion, lpa=standard_1005.lpa, rpa=standard_1005.rpa
)
fig = montage.plot(show=False, scale_factor=0)
plt.savefig('locations.pdf')
plt.close(fig)
