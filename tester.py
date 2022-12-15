from bs4 import BeautifulSoup as bs
from urllib.request import (
    urlopen, urlparse, urlunparse, urlretrieve)
import os
import sys
import re
from time import sleep
from os import listdir
import cohere
from cohere.classify import Example
import numpy as np
import college_filter as clf
from tqdm import tqdm

