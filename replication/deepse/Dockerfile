# Deep-SE (Tawosi et al. replication package) runtime: Python 2.7.18 + Keras 1.0.6 + Theano 0.9.0
# Built for linux/amd64 (the py2.7 scientific wheels below have no linux/aarch64 builds); on Apple
# silicon this runs under Rosetta:  docker build --platform linux/amd64 -t deepse:py27 .
# The code and data are NOT copied into the image; run_deepse.sh bind-mounts this directory at /work.
FROM python:2.7.18-slim-buster

# Debian buster is EOL: point apt at the archive.
RUN echo 'deb http://archive.debian.org/debian buster main' > /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends g++ perl libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Versions from Deep-SE/requirements.txt; scipy/pyyaml/six are unpinned transitive deps upstream,
# pinned here to the last Python-2 releases so the image is reproducible.
RUN pip install --no-cache-dir numpy==1.16.6 \
 && pip install --no-cache-dir scipy==1.2.3 six==1.16.0 PyYAML==5.4.1 \
 && pip install --no-cache-dir pandas==0.24.2 scikit-learn==0.18.2 h5py==2.7.1 theano==0.9.0 \
 && pip install --no-cache-dir --no-deps keras==1.0.6

RUN mkdir -p /root/.keras \
 && echo '{"backend": "theano", "floatx": "float32", "epsilon": 1e-07, "image_dim_ordering": "th"}' > /root/.keras/keras.json

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    THEANO_FLAGS=floatX=float32,device=cpu,blas.ldflags=-lopenblas,base_compiledir=/work/.theano

WORKDIR /work/Deep-SE
