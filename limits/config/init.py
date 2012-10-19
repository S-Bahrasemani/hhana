#!/usr/bin/env python

import os

HERE = os.path.dirname(os.path.abspath(__file__))

categories = ['ggf', 'boosted', 'vbf']
fitmethod = 'trackfit'
masses = range(100, 155, 5)
lumi = '1.'
lumi_rel_err = '0.039'
workspace = 'SYSTEMATICS'


def read_template(name):

    print "reading %s ..." % name
    return ''.join(open(os.path.join(HERE, name), 'r').readlines())


def write_config(name, content, context):

    name = name % context
    content = content % context
    print "writing %s ..." % name
    f = open(os.path.join(HERE, name), 'w')
    f.write(content)
    f.close()


channels = ('hh', 'elh', 'mulh')
channel_templates = {}
for channel in channels:
    channel_templates[channel] = read_template(
            'channel_%s.template' % channel)

comb_channels = ('hh', 'lh')
comb_templates = {}
comb_category_templates = {}
for channel in comb_channels:
    comb_templates[channel] = read_template(
            'combination_%s.template' % channel)
    comb_category_templates[channel] = read_template(
            'combination_category_%s.template' % channel)

full_comb_template = read_template(
        'combination_comb.template')
full_comb_category_template = read_template(
        'combination_category_comb.template')

for mass in masses:
    for channel in channels:
        for category in categories:
            write_config('%(channel)s_channel_%(category)s_%(mass)d.xml',
                channel_templates[channel], locals())

    for channel in comb_channels:
        write_config('%(channel)s_combination_%(mass)d.xml',
            comb_templates[channel], locals())

        for category in categories:
            write_config(
                '%(channel)s_combination_%(category)s_%(mass)d.xml',
                comb_category_templates[channel], locals())

    write_config('comb_combination_%(mass)d.xml',
            full_comb_template, locals())

    for category in categories:
        write_config('comb_combination_%(category)s_%(mass)d.xml',
            full_comb_category_template, locals())