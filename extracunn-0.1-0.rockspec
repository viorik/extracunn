package = "extracunn"
version = "0.1-0"

source = {
   url = "git://github.com/viorik/extracunn",
   tag = "master"
}

description = {
   summary = "Spatial convolutional no bias and Huber",
   detailed = [[
   Spatial convolutional no bias and Huber
   ]],
   homepage = "https://github.com/viorik/extracunn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "xlua >= 1.0",
   "cutorch"
}

build = {
   type = "command",
   build_command = [[
   		 cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)   ]],
   install_command = [[cd build && $(MAKE) install]]
}
