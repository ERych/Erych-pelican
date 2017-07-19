#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = u'Elizabeth Rychlinski'
SITENAME = u"Elizabeth's Portfolio"
SITEURL = 'https://erych.github.io'

RELATIVE_URLS = True

PATH = 'content'
PAGE_PATHS = ['pages']
STATIC_PAGES = ['images']

PLUGIN_PATH = ['plugins']
PLUGINS = ['ipynb.markup']
IPYNB_USE_META_SUMMARY = True

THEME = "themes/pure-single"
COVER_IMG_URL = "https://static.pexels.com/photos/205324/pexels-photo-205324.jpeg"
TAGLINE = "Experiments in Data Science"

SOCIAL = (('github', "https://github.com/erych"),
          ('linkedin-square', "https://www.linkedin.com/in/elizabeth-rychlinski-54a50a49/")
          )

TIMEZONE = 'EST'

DEFAULT_LANG = 'English'



# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

USE_FOLDER_AS_CATEGORY = True
FILENAME_METADATA = '(?P<date>\d{4}-\d{2}-\d{2})_(?P<slug>.*)'
DEFAULT_DATE = 'fs'
# Set some defaults§
DEFAULT_CATEGORY = 'projects'
DEFAULT_DATE_FORMAT = '%a %d %B %Y'
DEFAULT_PAGINATION = 10

MENUITEMS = [
    ('View Work', '/category/projects.html')
]
# Dynamic menu entries§
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_PAGES_ON_MENU = True

MARKUP = ('md', 'ipynb')
