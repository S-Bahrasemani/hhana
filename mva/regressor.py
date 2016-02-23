""" This module provides functions to retrive a trained regressor and
plot the predicted mass.
""" 

# python imports
import os
import pickle
import types
import shutil
from cStringIO import StringIO
from prettytable import PrettyTable


# rootpy imports
import ROOT
from rootpy.tree import Cut
from rootpy.plotting import Hist, HistStack, Legend, Canvas
from rootpy.plotting.style import set_style
# root_numpy imports
from root_numpy import rec2array, fill_hist

# scikit-learn imports
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score


# local imports
from . import log; log = log[__name__]
from . import MMC_MASS, MMC_PT
from . import variables, CACHE_DIR, BRT_DIR
from .systematics import systematic_name

set_style('ATLAS', shape='rect')
# used features for training BRT
brt_features = ['ditau_dr',
            'met_et',
            'ditau_scal_sum_pt',
            'ditau_vect_sum_pt',
            'ditau_mt_lep0_met',
            'ditau_mt_lep1_met',
            'ditau_dpt',
            'ditau_vis_mass',
            'ditau_dphi',
            'ditau_tau0_pt',
            'ditau_tau1_pt',
            'ditau_met_min_dphi',
            'ditau_met_lep0_cos_dphi',
            'ditau_met_lep1_cos_dphi',
            #'parent_m'
            ]

variables={
    'brt': {
        'title': 'BRT Mass',
        'root': 'BRT Mass',
        'filename': 'brt',
        'binning': (40,0, 200),
        'units':'GeV'
        }
    }

    
def get_regressor(train_id,
                  meta_name ='',
                  level='reco',
                  mode ='mix',
                  channel='hh'):
    """ Load a trained regression-tree model.
    
    Parameters
    ----------
    train_id: a unique number used for training in formate monthdayyear.

    meta_name: the name of meta classifiers, used as a lable to save the model to disk.

    level: level of training options are reco, truth.

    mode: trainig category of the samples; gg, vbf, mix.

    channel : training channel.

    Returns
    _______
    regressor: saved to the disk classifier
    
    """
    use_cache = True
    # attempt to load existing classifiers
    clf_filename = os.path.join(BRT_DIR,
                                str(train_id) +'_' +meta_name + '_{0}_{1}_{2}.pkl'.format(level, mode,channel
                                                                                          ))
    if os.path.isfile(clf_filename):
        # use a previously trained classifier
        log.info("found existing classifier in %s" % clf_filename)
        with open(clf_filename, 'r') as f:
            clf = pickle.load(f)
            out = StringIO()
        print >> out
        print >> out, clf
        log.info(out.getvalue())
            #print_feature_ranking(clf, self.fields)
    else:
        log.warning("could not open %s" % clf_filename)
        use_cache = False
    if use_cache:
        log.info("using previously trained classifiers")
        return clf
    else:
        log.error(
            "unable to load previously trained "
            "classifiers; train new ones")
        return 

def draw_brt (analysis, regressor, target_region='OS_ISOL', cat_defs=['presel', 'mva'], **kwargs):
    """ Draw the BRT mass.
    
    Parameters
    __________
    analysis: Analysis object
    
    target_region: the region that we are interested in 
    
    cat_defs : a list of categories to apply sepecific cuts
    
    regressor: Regressor; trained model(inherited from sklearn ensemble classes) object; loaded from disk
    
    Returns
    _______
    None: just creates plots and saves them to the disk.
    """
    
    for category in analysis.iter_categories(*cat_defs):

        samples = [
            analysis.qcd,
            analysis.others,
            analysis.ztautau
            ]
        hists = []
        for sample in samples + [analysis.data]:
            h = Hist(
                13, 0, 260, 
                name=sample.name, 
                title=sample.label, 
                **sample.hist_decor)
            rec = sample.merged_records(
                category, region=target_region, brt=regressor)
            fill_hist(h, rec['brt'], rec['weight'])
            hists.append(h)

        # stack all but data
        stack = HistStack(hists[:-1])
        c = Canvas()
        stack.Draw('hist')
        hists[-1].Draw('same')     
        leg = Legend(hists, pad=c, textsize=22)
        leg.Draw('same')
        c.SaveAs("./plots/brt_mass_{0}.png".format(category.name))
        c.Close()
