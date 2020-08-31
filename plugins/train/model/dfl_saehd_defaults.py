#!/usr/bin/env python3
"""
    The default options for the faceswap Dfl_SAEHD Model plugin.

    Defaults files should be named <plugin_name>_defaults.py
    Any items placed into this file will automatically get added to the relevant config .ini files
    within the faceswap/config folder.

    The following variables should be defined:
        _HELPTEXT: A string describing what this plugin does
        _DEFAULTS: A dictionary containing the options, defaults and meta information. The
                   dictionary should be defined as:
                       {<option_name>: {<metadata>}}

                   <option_name> should always be lower text.
                   <metadata> dictionary requirements are listed below.

    The following keys are expected for the _DEFAULTS <metadata> dict:
        datatype:  [required] A python type class. This limits the type of data that can be
                   provided in the .ini file and ensures that the value is returned in the
                   correct type to faceswap. Valid datatypes are: <class 'int'>, <class 'float'>,
                   <class 'str'>, <class 'bool'>.
        default:   [required] The default value for this option.
        info:      [required] A string describing what this option does.
        choices:   [optional] If this option's datatype is of <class 'str'> then valid
                   selections can be defined here. This validates the option and also enables
                   a combobox / radio option in the GUI.
        gui_radio: [optional] If <choices> are defined, this indicates that the GUI should use
                   radio buttons rather than a combobox to display this option.
        min_max:   [partial] For <class 'int'> and <class 'float'> datatypes this is required
                   otherwise it is ignored. Should be a tuple of min and max accepted values.
                   This is used for controlling the GUI slider range. Values are not enforced.
        rounding:  [partial] For <class 'int'> and <class 'float'> datatypes this is
                   required otherwise it is ignored. Used for the GUI slider. For floats, this
                   is the number of decimal places to display. For ints this is the step size.
        fixed:     [optional] [train only]. Training configurations are fixed when the model is
                   created, and then reloaded from the state file. Marking an item as fixed=False
                   indicates that this value can be changed for existing models, and will override
                   the value saved in the state file with the updated value in config. If not
                   provided this will default to True.
"""


_HELPTEXT = "DFL SAEHD Model (Adapted from https://github.com/iperov/DeepFaceLab)"


_DEFAULTS = dict(
    input_size=dict(
        default=128,
        info="Resolution (in pixels) of the input image to train on.\n"
             "BE AWARE Larger resolution will dramatically increase VRAM requirements.\n"
             "\nMust be divisible by 16.",
        datatype=int,
        rounding=16,
        min_max=(64, 640),
        group="size",
        fixed=True),
    architecture=dict(
        default="df",
        info="Model architecture:"
             "\n\t'df': Keeps the faces more natural."
             "\n\t'liae': Can help fix overly different face shapes.",
        datatype=str,
        choices=["df", "liae"],
        gui_radio=True,
        fixed=True,
        group="architecture"),
    dense_norm=dict(
        default=False,
        info="Add a Dense Normalization layer to the model. This is the same as adding '-u' in "
             "the original implementation. May increase likeness of the face.",
        datatype=bool,
        fixed=True,
        group="architecture"),
    true_face_power=dict(
        default=0,
        info="[df only] (Experimental). Discriminates the Original face to be more like the Swap "
             "face. Higher values mean more discrimination. A Typical value is '1'. Set to 0 to "
             "disable.",
        datatype=int,
        rounding=1,
        min_max=(0, 100),
        group="architecture",
        fixed=True),
    gan_power=dict(
        default=0,
        info="Train the network in Generative Adversarial manner. Attempts to mimic finer facial "
             " texture details. Should only be enabled when the face model is already well "
             "and you are looking to finalize the swap. Once this option has been enabled it "
             "should not then be disabled again. A Typical value is '1'. Set to '0' to disable.",
        datatype=int,
        rounding=1,
        min_max=(0, 100),
        group="architecture",
        fixed=True),
    face_style_power=dict(
        default=0.0,
        info="Attempt to learn colouring from the original face to apply to the swapped face.\n"
             "NB: Do not enable this option until you are at least 10,000 iterations in, and "
             "the faces look recognizable. Start with a low value (like 0.001) and adjust while "
             "checking progress.\n"
             "Warning: Enabling this option increases the chance of model collapse.",
        datatype=float,
        rounding=3,
        min_max=(0.0, 100.0),
        group="loss",
        fixed=False),
    bg_style_power=dict(
        default=0,
        info="Attempt to apply the area outside of the mask from the original face to the swapped "
             "face.\n"
             "This attempts to make the final face more like the swapped face.\n"
             "Warning: Enabling this option increases the chance of model collapse. A typical "
             "value is 2",
        datatype=int,
        rounding=2,
        min_max=(0, 100),
        group="loss",
        fixed=False),
    res_double=dict(
        default=False,
        info="(Experimental). Double the resolution using the same computational cost. This is "
             "the same as adding '-d' in the original implementation",
        datatype=bool,
        fixed=True,
        group="architecture"),
    autoencoder_dims=dict(
        default=256,
        info="Face information is stored in AutoEncoder dimensions. If there are not enough "
             "dimensions then certain facial features may not be recognized."
             "\nHigher number of dimensions are better, but require more VRAM.",
        datatype=int,
        rounding=32,
        min_max=(32, 1024),
        fixed=True,
        group="network"),
    encoder_dims=dict(
        default=64,
        info="Encoder dimensions per channel. Higher number of encoder dimensions will help the "
             "model to recognize more facial features, but will require more VRAM.",
        datatype=int,
        rounding=2,
        min_max=(16, 256),
        fixed=True,
        group="network"),
    decoder_dims=dict(
        default=64,
        info="Decoder dimensions per channel. Higher number of decoder dimensions will help the "
             "model to improve details, but will require more VRAM.",
        datatype=int,
        rounding=2,
        min_max=(16, 256),
        fixed=True,
        group="network"),
    clipnorm=dict(
        default=False,
        info="Controls gradient clipping of the optimizer. Can prevent model corruption at the "
             "expense of VRAM.",
        datatype=bool,
        fixed=False,
        group="settings"))
