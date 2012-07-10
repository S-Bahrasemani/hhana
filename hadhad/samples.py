import os
import sys
import atexit
from operator import add
import math

import numpy as np
from numpy.lib import recfunctions

# for reproducibilty
np.random.seed(1987) # my birth year ;)

from higgstautau.hadhad.periods import total_lumi
from higgstautau import datasets
from higgstautau.decorators import cached_property, memoize_method
from higgstautau import samples as samples_db
from higgstautau import xsec

# Higgs cross sections
import yellowhiggs

from rootpy.plotting import Hist, Canvas, HistStack
from rootpy.io import open as ropen
from rootpy.tree import Tree, Cut
from rootpy.utils import asrootpy
from rootpy import root2array as r2a
from rootpy.math.stats.correlation import correlation_plot

import categories
import features
from systematics import iter_systematics


NTUPLE_PATH = os.getenv('HIGGSTAUTAU_NTUPLE_DIR')
if not NTUPLE_PATH:
    sys.exit("You did not source setup.sh")
NTUPLE_PATH = os.path.join(NTUPLE_PATH, 'hadhad')
PROCESSOR = 'HHProcessor'
TOTAL_LUMI = total_lumi()
TAUTAUHADHADBR = 0.412997
VERBOSE = False
DB = datasets.Database(name='datasets_hh', verbose=VERBOSE)
FILES = {}
WORKING_POINT = 'Tight'
ID = Cut('tau1_JetBDTSig%s==1 && tau2_JetBDTSig%s==1' %
         (WORKING_POINT, WORKING_POINT))
NOID = Cut('tau1_JetBDTSig%s!=1 && tau2_JetBDTSig%s!=1' %
           (WORKING_POINT, WORKING_POINT))
OS = Cut('tau1_charge * tau2_charge == -1')
NOT_OS = Cut('tau1_charge * tau2_charge != -1')
SS = Cut('tau1_charge * tau2_charge == 1')
# mass_jet1_jet2 > 100000
TEMPFILE = ropen('tempfile.root', 'recreate')


#@atexit.register
def cleanup():

    TEMPFILE.Close()
    os.unlink(TEMPFILE.GetName())


def std(X):

    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def correlations(signal, signal_weight,
                 background, background_weight,
                 branches, channel):

    # draw correlation plots
    names = [features.VARIABLES[branch]['title'] for branch in branches]
    correlation_plot(signal, signal_weight, names,
                     "correlation_signal_%s" % channel)
    correlation_plot(background, background_weight, names,
                     "correlation_background_%s" % channel)


def get_samples(mass=None):

    mc_ztautau = MC_Ztautau()
    mc_ewk = MC_EWK()
    mc_ttbar = MC_TTbar()
    mc_singletop = MC_SingleTop()
    mc_diboson = MC_Diboson()

    backgrounds = (
        mc_ztautau,
        mc_ewk,
        mc_ttbar,
        mc_singletop,
        mc_diboson,
    )

    signals = (
        MC_VBF(mass=mass),
        MC_ggF(mass=mass),
        MC_WH(mass=mass),
        MC_ZH(mass=mass)
    )
    return signals, backgrounds


def make_classification(
        signals,
        backgrounds,
        category,
        region,
        branches,
        cuts=None,
        train_fraction=None,
        max_sig_train=None,
        max_bkg_train=None,
        max_sig_test=None,
        max_bkg_test=None,
        norm_sig_to_bkg_train=True,
        norm_sig_to_bkg_test=False,
        same_size_train=True,
        same_size_test=False,
        standardize=False,
        systematic=None):

    signal_train_arrs = []
    signal_weight_train_arrs = []
    signal_test_arrs = []
    signal_weight_test_arrs = []

    for signal in signals:
        train, test = signal.train_test(
            category, region,
            branches=branches,
            train_fraction=train_fraction,
            cuts=cuts,
            systematic=systematic)
        signal_weight_train_arrs.append(train['weight'])
        signal_weight_test_arrs.append(test['weight'])

        signal_train_arrs.append(
            np.vstack(train[branch] for branch in branches).T)
        signal_test_arrs.append(
            np.vstack(test[branch] for branch in branches).T)

    background_train_arrs = []
    background_weight_train_arrs = []
    background_test_arrs = []
    background_weight_test_arrs = []

    for background in backgrounds:
        train, test = background.train_test(
            category, region,
            branches=branches,
            train_fraction=train_fraction,
            cuts=cuts,
            systematic=systematic)
        background_weight_train_arrs.append(train['weight'])
        background_weight_test_arrs.append(test['weight'])

        background_train_arrs.append(
            np.vstack(train[branch] for branch in branches).T)
        background_test_arrs.append(
            np.vstack(test[branch] for branch in branches).T)

    signal_train = np.concatenate(signal_train_arrs)
    signal_weight_train = np.concatenate(signal_weight_train_arrs)
    signal_test = np.concatenate(signal_test_arrs)
    signal_weight_test = np.concatenate(signal_weight_test_arrs)

    background_train = np.concatenate(background_train_arrs)
    background_weight_train = np.concatenate(background_weight_train_arrs)
    background_test = np.concatenate(background_test_arrs)
    background_weight_test = np.concatenate(background_weight_test_arrs)

    if max_sig_train is not None and max_sig_train < len(signal_train):
        subsample = np.random.permutation(max_sig_train)[:len(signal_train)]
        signal_train = signal_train[subsample]
        signal_weight_train = signal_weight_train[subsample]

    if max_bkg_train is not None and max_bkg_train < len(background_train):
        subsample = np.random.permutation(max_bkg_train)[:len(background_train)]
        background_train = background_train[subsample]
        background_weight_train = background_weight_train[subsample]

    if max_sig_test is not None and max_sig_test < len(signal_test):
        subsample = np.random.permutation(max_sig_test)[:len(signal_test)]
        signal_test = signal_test[subsample]
        signal_weight_test = signal_weight_test[subsample]

    if max_bkg_test is not None and max_bkg_test < len(background_test):
        subsample = np.random.permutation(max_bkg_test)[:len(background_test)]
        background_test = background_test[subsample]
        background_weight_test = background_weight_test[subsample]

    if same_size_train:
        if len(background_train) > len(signal_train):
            # random subsample of background so it's the same size as signal
            subsample = np.random.permutation(
                len(background_train))[:len(signal_train)]
            background_train = background_train[subsample]
            background_weight_train = background_weight_train[subsample]
        elif len(background_train) < len(signal_train):
            # random subsample of signal so it's the same size as background
            subsample = np.random.permutation(
                len(signal_train))[:len(background_train)]
            signal_train = signal_train[subsample]
            signal_weight_train = signal_weight_train[subsample]

    if same_size_test:
        if len(background_test) > len(signal_test):
            # random subsample of background so it's the same size as signal
            subsample = np.random.permutation(
                len(background_test))[:len(signal_test)]
            background_test = background_test[subsample]
            background_weight_test = background_weight_test[subsample]
        elif len(background_test) < len(signal_test):
            # random subsample of signal so it's the same size as background
            subsample = np.random.permutation(
                len(signal_test))[:len(background_test)]
            signal_test = signal_test[subsample]
            signal_weight_test = signal_weight_test[subsample]

    if norm_sig_to_bkg_train:
        # normalize signal to background
        signal_weight_train *= (
            background_weight_train.sum() / signal_weight_train.sum())

    if norm_sig_to_bkg_test:
        # normalize signal to background
        signal_weight_test *= (
            background_weight_test.sum() / signal_weight_test.sum())

    print "Training Samples:"
    print "Signal: %d events, %s features" % signal_train.shape
    print "Sum(signal weights): %f" % signal_weight_train.sum()
    print "Background: %d events, %s features" % background_train.shape
    print "Sum(background weight): %f" % background_weight_train.sum()
    print
    print "Test Samples:"
    print "Signal: %d events, %s features" % signal_test.shape
    print "Sum(signal weights): %f" % signal_weight_test.sum()
    print "Background: %d events, %s features" % background_test.shape
    print "Sum(background weight): %f" % background_weight_test.sum()

    # create training/testing samples
    sample_train = np.concatenate((background_train, signal_train))
    sample_test = np.concatenate((background_test, signal_test))

    sample_weight_train = np.concatenate(
        (background_weight_train, signal_weight_train))
    sample_weight_test = np.concatenate(
        (background_weight_test, signal_weight_test))

    if standardize:
        sample_train = std(sample_train)
        sample_test = std(sample_test)

    labels_train = np.concatenate(
        (np.zeros(len(background_train)), np.ones(len(signal_train))))
    labels_test = np.concatenate(
        (np.zeros(len(background_test)), np.ones(len(signal_test))))

    # random permutation of training sample
    perm = np.random.permutation(len(labels_train))
    sample_train = sample_train[perm]
    sample_weight_train = sample_weight_train[perm]
    labels_train = labels_train[perm]

    # split the dataset in two equal parts respecting label proportions
    #train, test = iter(StratifiedKFold(labels, 2)).next()
    return sample_train, sample_test,\
        sample_weight_train, sample_weight_test,\
        labels_train, labels_test


class Sample(object):

    REGIONS = {
        'ALL': Cut(),
        'OS': OS,
        '!OS': NOT_OS,
        'SS': SS,
        'OS-ID': OS & ID,
        '!OS-ID': NOT_OS & ID,
        'SS-ID': SS & ID,
        'OS-NOID': OS & NOID,
        '!OS-NOID': NOT_OS & NOID,
        'SS-NOID': SS & NOID}

    CATEGORIES = dict([
        (name, Cut('category==%d' % info['code']))
        if info['code'] is not None
        else (name, Cut(''))
        for name, info in categories.CATEGORIES.items()])

    def __init__(self, scale=1., cuts=None):

        self.scale = scale
        if cuts is None:
            self._cuts = Cut()
        else:
            self._cuts = cuts

    def cuts(self, category, region):

        return (Sample.CATEGORIES[category] &
                categories.CATEGORIES[category]['cuts'] &
                Sample.REGIONS[region] & self._cuts)

    def train_test(self,
                   category,
                   region,
                   branches,
                   train_fraction=None,
                   cuts=None,
                   systematic=None):
        """
        Return recarray for training and for testing
        """
        if train_fraction is not None:
            assert 0 < train_fraction < 1.
            if isinstance(self, MC):
                branches = branches + [
                    'mc_weight',
                    'pileup_weight',
                    'tau1_weight',
                    'tau2_weight']

            train_arrs = []
            test_arrs = []

            for tree in self.trees(
                    category,
                    region,
                    cuts=cuts,
                    systematic=systematic):
                arr = r2a.tree_to_recarray(
                    tree,
                    branches=branches,
                    include_weight=True,
                    weight_name='weight')
                if isinstance(self, MC):
                    # merge the three weight columns
                    arr['weight'] *= (arr['mc_weight'] * arr['pileup_weight'] *
                                      arr['tau1_weight'] * arr['tau2_weight'])
                    # remove the mc_weight and pileup_weight fields
                    arr = recfunctions.rec_drop_fields(
                        arr,
                        ['mc_weight',
                         'pileup_weight',
                         'tau1_weight',
                         'tau2_weight'])
                split_idx = int(train_fraction * float(arr.shape[0]))
                arr_train, arr_test = arr[:split_idx], arr[split_idx:]
                # scale the weights to account for train_fraction
                arr_train['weight'] *= 1. / train_fraction
                arr_test['weight'] *= 1. / (1. - train_fraction)
                train_arrs.append(arr_train)
                test_arrs.append(arr_test)
            arr_train, arr_test = np.hstack(train_arrs), np.hstack(test_arrs)
        else:
            arr = self.recarray(
                category,
                region,
                branches,
                include_weight=True,
                cuts=cuts,
                systematic=systematic)
            arr_train, arr_test = arr, arr

        return arr_train, arr_test

    def recarray(self,
                 category,
                 region,
                 branches,
                 include_weight=True,
                 cuts=None,
                 systematic=None):

        if include_weight and isinstance(self, MC):
            branches = branches + [
                'mc_weight',
                'pileup_weight',
                'tau1_weight',
                'tau2_weight']

        try:
            arr = r2a.tree_to_recarray(
                self.trees(
                    category,
                    region,
                    cuts=cuts,
                    systematic=systematic),
                branches=branches,
                include_weight=include_weight,
                weight_name='weight')
        except IOError, e:
            raise IOError("%s: %s" % (self.__class__.__name__, e))

        if include_weight and isinstance(self, MC):
            # merge the three weight columns
            arr['weight'] *= (arr['mc_weight'] * arr['pileup_weight'] *
                              arr['tau1_weight'] * arr['tau2_weight'])
            # remove the mc_weight and pileup_weight fields
            arr = recfunctions.rec_drop_fields(
                arr,
                ['mc_weight',
                 'pileup_weight',
                 'tau1_weight',
                 'tau2_weight'])
        return arr

    def ndarray(self,
                category,
                region,
                branches,
                include_weight=True,
                cuts=None,
                systematic=None):

        return r2a.recarray_to_ndarray(
                   self.recarray(
                       category,
                       region,
                       branches=branches,
                       include_weight=include_weight,
                       cuts=cuts,
                       systematic=systematic))


class Data(Sample):

    DATA_FILE = ropen('.'.join([os.path.join(NTUPLE_PATH, PROCESSOR),
                                'data.root']))

    def __init__(self, cuts=None):

        super(Data, self).__init__(scale=1., cuts=cuts)
        self.data = self.DATA_FILE.Get('higgstautauhh')
        self.label = '2011 Data $\sqrt{s} = 7$ TeV\n$\int L dt = 4.7$ fb$^{-1}$'
        self.name = 'Data'

    def draw(self, expr, category, region, bins, min, max, cuts=None):

        hist = Hist(bins, min, max, title=self.label, name=self.name)
        self.draw_into(hist, expr, category, region, cuts=cuts)
        return hist

    def draw_into(self, hist, expr, category, region, cuts=None):

        self.data.draw(expr, self.cuts(category, region) & cuts, hist=hist)

    def trees(self,
              category,
              region,
              cuts=None,
              systematic=None):
        """
        systematics do not apply to data but the argument is present for
        coherence with the other samples
        """
        TEMPFILE.cd()
        return [asrootpy(self.data.CopyTree(self.cuts(category, region) & cuts))]


class MC(Sample):

    def __init__(self, scale=1., cuts=None):

        super(MC, self).__init__(scale=scale, cuts=cuts)
        self.datasets = []

        for i, name in enumerate(self.samples):

            ds = DB[name]
            trees = {}
            weighted_events = {}

            for sys_object, sys_type, sys_variations in \
                iter_systematics('hadhad', include_nominal=True):

                if sys_object is None:
                    # nominal
                    sys_terms = ('NOMINAL',)
                    trees['NOMINAL'] = None
                    weighted_events['NOMINAL'] = None
                else:
                    sys_terms = [sys_type + '_' + v for v in sys_variations]
                    trees[sys_type] = {}
                    weighted_events[sys_type] = {}
                    for v in sys_variations:
                        trees[sys_type][v] = None
                        weighted_events[sys_type][v] = None

                for sys_term in sys_terms:
                    if ds.name in FILES and sys_term in FILES[ds.name]:
                        rfile = FILES[ds.name][sys_term]
                    else:
                        if sys_term == 'NOMINAL':
                            rfile = ropen('.'.join([
                                os.path.join(NTUPLE_PATH, PROCESSOR), ds.name, 'root']))
                            trees[sys_term] = rfile.Get('higgstautauhh')
                            weighted_events[sys_term] = rfile.cutflow[1]
                        else:
                            sys_type, variation = sys_term.split('_')
                            rfile = ropen('.'.join([
                                os.path.join(NTUPLE_PATH, PROCESSOR),
                                '_'.join([ds.name, sys_term]), 'root']))
                            trees[sys_type][variation] = rfile.Get('higgstautauhh')
                            weighted_events[sys_type][variation] = rfile.cutflow[1]
                        if ds.name not in FILES:
                            FILES[ds.name] = {}
                        FILES[ds.name][sys_term] = rfile

                if isinstance(self, MC_Higgs):
                    # use yellowhiggs for cross sections
                    xs = yellowhiggs.xsbr(
                            7, self.mass[i],
                            self.mode, 'tautau')[0] * TAUTAUHADHADBR
                    kfact = 1.
                    effic = 1.
                else:
                    # use xsec for cross sections
                    xs, kfact, effic = xsec.xsec_kfact_effic('lephad', ds.id)
                if VERBOSE:
                    print ds.name, xs, kfact, effic
                    #print tree.GetEntries(), weighted_events
            print trees
            print weighted_events
            self.datasets.append((ds, trees, weighted_events, xs, kfact, effic))

    @property
    def label(self):

        l = self._label
        if self.scale != 1. and not isinstance(self, MC_Ztautau):
            l += r' ($\sigma_{SM} \times %g$)' % self.scale
        return l

    def draw(self, expr, category, region, bins, min, max, cuts=None):

        hist = Hist(bins, min, max, title=self.label, name=self.name)
        self.draw_into(hist, expr, category, region, cuts=cuts)
        return hist

    def draw_into(self, hist, expr, category, region, cuts=None):

        selection = self.cuts(category, region) & cuts
        if isinstance(expr, (list, tuple)):
            exprs = expr
        else:
            exprs = (expr,)

        sys_hists = {}
        sys_hist = hist.Clone()
        sys_hist.Reset()

        for ds, sys_trees, sys_events, xs, kfact, effic in self.datasets:
            weight = TOTAL_LUMI * self.scale * xs * kfact * effic / events
            weighted_selection = ('%.5f * mc_weight * pileup_weight * '
                                  'tau1_weight * tau2_weight * (%s)' %
                                  (weight, selection))
            if VERBOSE:
                print weighted_selection

            for sys_type, variations in sys_trees.items():
                pass


            for expr in exprs:
                tree.Draw(expr, weighted_selection, hist=hist)

        # set the systematics
        hist.systematics = sys_hists

    def trees(self, category, region, cuts=None,
              systematic=None):

        TEMPFILE.cd()
        selection = self.cuts(category, region) & cuts
        trees = []
        if systematic is not None:
            sys_type, sys_var = systematic.split('_')
        for ds, sys_trees, sys_events, xs, kfact, effic in self.datasets:
            if systematic is None:
                tree = sys_trees['NOMINAL']
                events = sys_events['NOMINAL']
            else:
                tree = sys_trees[sys_type][sys_var]
                events = sys_events[sys_type][sys_var]
            weight = TOTAL_LUMI * self.scale * xs * kfact * effic / events
            selected_tree = asrootpy(tree.CopyTree(selection))
            selected_tree.SetWeight(weight)
            trees.append(selected_tree)
        return trees


class MC_Ztautau(MC):

    def __init__(self, scale=1., cuts=None):
        """
        Instead of setting the k factor here
        the normalization is determined by a fit to the data
        """
        yml = samples_db.BACKGROUNDS['hadhad']['ztautau']
        self.name = 'Ztautau'
        self._label = yml['latex']
        self.samples = yml['samples']
        super(MC_Ztautau, self).__init__(scale=scale,
                                         cuts=cuts)


class MC_EWK(MC):

    def __init__(self, scale=1., cuts=None):

        yml = samples_db.BACKGROUNDS['hadhad']['ewk']
        self.name = 'EWK'
        self._label = yml['latex']
        self.samples = yml['samples']
        super(MC_EWK, self).__init__(scale=scale,
                                     cuts=cuts)


class MC_Top(MC):

    def __init__(self, scale=1., cuts=None):

        yml = samples_db.BACKGROUNDS['hadhad']['top']
        self.name = 'Top'
        self._label = yml['latex']
        self.samples = yml['samples']
        super(MC_Top, self).__init__(scale=scale,
                                           cuts=cuts)


class MC_Diboson(MC):

    def __init__(self, scale=1., cuts=None):

        yml = samples_db.BACKGROUNDS['hadhad']['diboson']
        self.name = 'Diboson'
        self._label = yml['latex']
        self.samples = yml['samples']
        super(MC_Diboson, self).__init__(scale=scale,
                                         cuts=cuts)


class MC_Higgs(MC):

    MASS_POINTS = range(100, 155, 5)

    MODES = {
        'ggH': 'ggf',
        'VBFH': 'vbf',
        'ZH': 'zh',
        'WH': 'wh',
    }

    def __init__(self, mode, generator, mass=None, scale=1., cuts=None):

        self.mode = MC_Higgs.MODES[mode]

        if mass is None:
            mass = MC_Higgs.MASS_POINTS

        if isinstance(mass, (list, tuple)):
            self._label = r'%s $H\rightarrow\tau_{h}\tau_{h}$' % mode
            self.name = 'Signal'
            self.mass = mass
        else:
            self._label = r'%s $H(%d)\rightarrow\tau_{h}\tau_{h}$' % \
                           (mode, mass)
            self.name = 'Signal%d' % mass
            self.mass = [mass]

        self.samples = ['%s%s%d_tautauhh.mc11c' % (generator, mode, m)
                        for m in self.mass]
        super(MC_Higgs, self).__init__(scale=scale,
                                       cuts=cuts)


class MC_VBF(MC_Higgs):

    def __init__(self, mass=None, scale=1., cuts=None):

        super(MC_VBF, self).__init__(
                mode='VBFH',
                mass=mass,
                generator='PowHegPythia_',
                scale=scale,
                cuts=cuts)


class MC_ggF(MC_Higgs):

    def __init__(self, mass=None, scale=1., cuts=None):

        super(MC_ggF, self).__init__(
                mode='ggH',
                mass=mass,
                generator='PowHegPythia_',
                scale=scale,
                cuts=cuts)


class MC_WH(MC_Higgs):

    def __init__(self, mass=None, scale=1., cuts=None):

        super(MC_WH, self).__init__(
                mode='WH',
                mass=mass,
                generator='Pythia',
                scale=scale,
                cuts=cuts)


class MC_ZH(MC_Higgs):

    def __init__(self, mass=None, scale=1., cuts=None):

        super(MC_ZH, self).__init__(
                mode='ZH',
                mass=mass,
                generator='Pythia',
                scale=scale,
                cuts=cuts)


class QCD(Sample):

    def __init__(self, data, mc, scale=1., sample_region='SS'):

        super(QCD, self).__init__(scale=scale)
        self.data = data
        self.mc = mc
        self.name = 'QCD'
        self.label = 'QCD'
        self.scale = 1.
        self.sample_region = sample_region

    def draw(self, expr, category, region, bins, min, max, cuts=None,
             sample_region=None):

        if sample_region is None:
            sample_region = self.sample_region

        hist = Hist(bins, min, max, title=self.label, name=self.name)
        self.draw_into(hist, expr, category, region, cuts=cuts,
                       sample_region=sample_region)
        return hist

    def draw_into(self, hist, expr, category, region, cuts=None,
                  sample_region=None):

        if sample_region is None:
            sample_region = self.sample_region

        MC_bkg_notOS = hist.Clone()
        for mc in self.mc:
            mc.draw_into(MC_bkg_notOS, expr, category, sample_region, cuts=cuts)

        # assume norm factor of 1., to be determined later in fit
        self.data.draw_into(hist, expr,
                            category, sample_region, cuts=cuts)
        hist -= MC_bkg_notOS
        hist *= self.scale
        hist.SetTitle(self.label)

    def scores(self,
               clf,
               category,
               region,
               branches,
               train_fraction,
               cuts=None,
               systematic=None):

        # SS data
        train, test = self.data.train_test(category=category,
                                           region=self.sample_region,
                                           branches=branches,
                                           train_fraction=train_fraction,
                                           cuts=cuts)
        weight = test['weight']
        sample = np.vstack(test[branch] for branch in branches).T
        scores = clf.predict_proba(sample)[:,-1]

        # subtract SS MC
        for mc in self.mc:
            # didn't train on MC here if using SS or !OS
            train, test = mc.train_test(category=category,
                                        region=self.sample_region,
                                        branches=branches,
                                        train_fraction=train_fraction,
                                        cuts=cuts,
                                        systematic=systematic)
            sample = np.vstack(test[branch] for branch in branches).T
            scores = np.concatenate((scores, clf.predict_proba(sample)[:,-1]))
            weight = np.concatenate((weight, test['weight'] * -1))

        weight *= self.scale
        return scores, weight

    def trees(self, category, region, cuts=None,
              systematic=None):

        TEMPFILE.cd()
        trees = [asrootpy(self.data.data.CopyTree(
                    self.data.cuts(category,
                                   region=self.sample_region) & cuts))]
        for mc in self.mc:
            _trees = mc.trees(
                    category,
                    region=self.sample_region,
                    cuts=cuts,
                    systematic=systematic)
            for tree in _trees:
                tree.Scale(-1)
            trees += _trees
        for tree in trees:
            tree.Scale(self.scale)
        return trees


if __name__ == '__main__':

    from pyAMI.query import print_table
    signals, backgrounds = get_samples('2jet', purpose='train')

    table = [['dataset',
              'mean sigma [pb]', 'min sigma [pb]', 'max sigma [pb]',
              'sigma factor',
              'filter effic', 'K factor']]
    format = "%s %.8f %.8f %.8f %.6f %.6f %.6f"
    for sample in signals + backgrounds:
        if isinstance(sample, QCD):
            continue
        for datasets in sample.datasets:
            for ds, tree, events, (xsec, xsec_min, xsec_max, effic) in datasets:
                row = format % (ds.ds, xsec*1E3, xsec_min*1E3, xsec_max*1E3, ds.xsec_factor, effic, 1.)
                table.append(row.split())
    print
    print
    table.sort(key=lambda row: row[0])
    print_table(table)
