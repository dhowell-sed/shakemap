#!/usr/bin/env python

import copy
from importlib import import_module
import logging

import numpy as np

from openquake.hazardlib.gsim.base import GMPE, IPE
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import (
    CampbellBozorgnia2014)
from openquake.hazardlib.imt import PGA, PGV, SA
from openquake.hazardlib import const

from shakelib.conversions.imt.newmark_hall_1982 import NewmarkHall1982
from shakelib.conversions.imc.boore_kishida_2017 import BooreKishida2017
from shakelib.sites import Sites


class GmpeRunner:
    """
    A GmpeRunner encapsulates all logic for executing a set of Gmpe models, a "GMPE Set" in ShakeMap terminology.

    Note: You can add logic to this class to handle non OpenQuake models if desired.

    The GmpeRunner should only be instantiated via one of the class factory methods

    gr = gmpe_runner.from_config(..)
    lnmu, lnstdev = gr.compute_mean_and_stdevs()
    ...

    """

    # The key to the merged branches in the
    COMBINED_GMPE_KEY = 'combined'

    def __init__(self, gmpes, weights,
                 imc=const.IMC.GREATER_OF_TWO_HORIZONTAL,
                 default_gmpes_for_site=None,
                 default_gmpes_for_site_weights=None,
                 reference_vs30=760):
        """
                Construct a GmpeRunner instance from lists of GMPEs and weights.

                Args:
                    reference_vs30:
                    default_gmpes_for_site_weights:
                    gmpes (list): List of OpenQuake
                        `GMPE <http://docs.openquake.org/oq-hazardlib/master/gsim/index.html#built-in-gsims>`__
                        instances.

                    weights (list): List of weights; must sum to 1.0.

                    imc: Requested intensity measure component. Must be one listed
                        `here <http://docs.openquake.org/oq-hazardlib/master/const.html?highlight=imc#openquake.hazardlib.const.IMC>`__.
                        The amplitudes returned by the GMPEs will be converted to this
                        IMT. Default is 'GREATER_OF_TWO_HORIZONTAL', which is used by
                        ShakeMap. See discussion in
                        `this section <http://usgs.github.io/shakemap/tg_choice_of_parameters.html#use-of-peak-values-rather-than-mean>`__
                        of the ShakeMap manual.

                    default_gmpes_for_site (list):
                        Optional list of OpenQuake GMPE instance to use as a site term
                        for any of the GMPEs that do not have a site term.

                        Notes:

                            * We do not check for consistency in the reference rock
                              definition, so the user nees to be aware of this issue and
                              holds responsibility for ensuring compatibility.
                            * We check whether or not a GMPE has a site term by c
                              hecking the REQUIRES_SITES_PARAMETERS slot for vs30.

                    default_gmpes_for_site_weights: Weights for default_gmpes_for_site.
                        Must sum to one and be same length as default_gmpes_for_site.
                        If None, then weights are set to be equal.

                    reference_vs30:
                        Reference rock Vs30 in m/s. We do not check that this matches
                        the reference rock in the GMPEs so this is the responsibility
                        of the user.

                """
        # Input validation
        if np.abs(np.sum(weights) - 1.0) > 1e-7:
            raise Exception('Weights must sum to one.')
        if len(weights) != len(gmpes):
            raise Exception(
                'Length of weights must match length of GMPE list.')
        for g in gmpes:
            if not isinstance(g, GMPE) and not isinstance(g, IPE):
                raise Exception("\"%s\" is not a GMPE or IPE instance." % g)

        self.GMPES = gmpes
        self.WEIGHTS = weights
        self.CUTOFF_DISTANCE = None
        self.WEIGHTS_LARGE_DISTANCE = None
        self.GMPE_LIMITS = None

        # ---------------------------------------------------------------------
        # Combine the intensity measure types. This is problematic:
        #   - Logically, we should only include the intersection of the sets
        #     of imts for the different GMPEs.
        #   - In practice, this is not feasible because most GMPEs in CEUS and
        #     subduction zones do not have PGV.
        #   - So instead we will use the union of the imts and then convert
        #     to get the missing imts later in get_mean_and_stddevs.
        # ---------------------------------------------------------------------

        imts = [g.DEFINED_FOR_INTENSITY_MEASURE_TYPES for g in gmpes]
        self.DEFINED_FOR_INTENSITY_MEASURE_TYPES = set.union(*imts)

        # ---------------------------------------------------------------------
        # For VirtualIPE class, we also want to know if ALL of the GMPEs are
        # defined for PGV, in which case we will convert from PGV to MI,
        # otherwise use PGA or Sa.
        # ---------------------------------------------------------------------
        haspgv = [PGV in g.DEFINED_FOR_INTENSITY_MEASURE_TYPES for g in gmpes]
        self.ALL_GMPES_HAVE_PGV = all(haspgv)

        # ---------------------------------------------------------------------
        # Store intensity measure types for conversion in get_mean_and_stddevs.
        # ---------------------------------------------------------------------
        self.IMCs = [g.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT for g in gmpes]

        # ---------------------------------------------------------------------
        # Store the component
        # ---------------------------------------------------------------------
        self.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = imc

        # ---------------------------------------------------------------------
        # Intersection of GMPE standard deviation types
        # ---------------------------------------------------------------------
        stdlist = [set(g.DEFINED_FOR_STANDARD_DEVIATION_TYPES) for g in gmpes]
        self.DEFINED_FOR_STANDARD_DEVIATION_TYPES = \
            set.intersection(*stdlist)

        # ---------------------------------------------------------------------
        # Need union of site parameters, but it is complicated by the
        # different depth parameter flavors.
        # ---------------------------------------------------------------------
        sitepars = [g.REQUIRES_SITES_PARAMETERS for g in gmpes]
        self.REQUIRES_SITES_PARAMETERS = set.union(*sitepars)

        # ---------------------------------------------------------------------
        # Construct a list of whether or not each GMPE has a site term
        # ---------------------------------------------------------------------
        self.HAS_SITE = ['vs30' in g.REQUIRES_SITES_PARAMETERS for g in gmpes]

        # ---------------------------------------------------------------------
        # Checks and sort out defaults
        # ---------------------------------------------------------------------

        # things to check if default_gmpes_for_site is provided
        if default_gmpes_for_site is not None:
            # check that default_gmpe_for_site are OQ GMPEs or None
            for g in default_gmpes_for_site:
                if not isinstance(g, GMPE):
                    raise Exception("\"%s\" is not a GMPE instance." % g)

            # apply default weights if necessary
            if default_gmpes_for_site_weights is None:
                n = len(default_gmpes_for_site)
                default_gmpes_for_site_weights = [1 / n] * n

        # Things to check if one or more GMPE does not have a site term
        if not all(self.HAS_SITE):
            # Raise an exception if no default site is provided
            if default_gmpes_for_site is None:
                raise Exception('Must provide default_gmpes_for_site if one or'
                                ' more GMPE does not have site term.')

            # If weights are unspecified, use equal weight
            if default_gmpes_for_site_weights is None:
                default_gmpes_for_site_weights = \
                    [1 / len(default_gmpes_for_site)] * \
                    len(default_gmpes_for_site)

            # check that length of default_gmpe_for_site matches length of
            # default_gmpe_for_site_weights
            if len(default_gmpes_for_site_weights) != \
                    len(default_gmpes_for_site):
                raise Exception('Length of default_gmpes_for_site_weights '
                                'must match length of default_gmpes_for_site '
                                'list.')

            # check weights sum to one if needed
            if not all(self.HAS_SITE):
                if np.sum(default_gmpes_for_site_weights) != 1.0:
                    raise Exception('default_gmpes_for_site_weights must sum'
                                    ' to one.')

        # Note: if ALL of the GMPEs do not have a site term (requiring Vs30),
        #       then REQUIRES_SITES_PARAMETERS for the MultiGMPE will not
        #       include Vs30 even though it will be needed to compute the
        #       default site term. So if the site checks have passed to this
        #       point, we should add Vs30 to the set of required site pars:
        self.REQUIRES_SITES_PARAMETERS = set.union(
            self.REQUIRES_SITES_PARAMETERS, set(['vs30']))

        self.DEFAULT_GMPES_FOR_SITE = default_gmpes_for_site
        self.DEFAULT_GMPES_FOR_SITE_WEIGHTS = default_gmpes_for_site_weights
        self.REFERENCE_VS30 = reference_vs30

        # ---------------------------------------------------------------------
        # Union of rupture parameters
        # ---------------------------------------------------------------------
        ruppars = [g.REQUIRES_RUPTURE_PARAMETERS for g in gmpes]
        self.REQUIRES_RUPTURE_PARAMETERS = set.union(*ruppars)

        # ---------------------------------------------------------------------
        # Union of distance parameters
        # ---------------------------------------------------------------------
        distpars = [g.REQUIRES_DISTANCES for g in gmpes]
        self.REQUIRES_DISTANCES = set.union(*distpars)

    @staticmethod
    def flatten_context_arrays(context):
        """
        Flatten all arrays in a OpenQuake BaseContext which are used in a GMPE.  This is a patch to avoid errors when
        passing multi-dimensional arrays to some GMPEs.

        Args:
            context (openquake.hazardlib.contexts.BaseContext): The OpenQuake context which will be passed to a GMPE

        Returns:
            old_shapes (dict): Mapping between context fields and their old np.array.shape of format
                e.g. {'some_field':(,100)}
        """
        old_shapes = {}
        for k, v in context.__dict__.items():
            if k == '_slots_':
                continue
            if (k is not 'lons') and (k is not 'lats') and v is not None:
                old_shapes[k] = v.shape
                context.__dict__[k] = np.reshape(context.__dict__[k], (-1,))

    @staticmethod
    def restore_context_array_shapes(context, old_shapes):
        """
        Restore the shape of all arrays in an OpenQuake BaseContext which were flattened with
        GmpeRunner.flatten_context_arrays()

        Args:
            context (openquake.hazardlib.contexts.BaseContext): The OpenQuake context which will be passed to a GMPE
            old_shapes (dict): Mapping between context fields and their old np.array.shape of format

        Returns:
            None

        """
        for k in old_shapes:
            context.__dict__[k] = np.reshape(context.__dict__[k], old_shapes[k])

    def _oq_gmpe_get_mean_and_stddevs(self, gmpe, has_site, sites, rup, dists, imt, stddev_types):
        """
        Protected method for calling an OpenQuake GMPE's get_mean_and_stddevs() function.

        This method encapsulates all logic for calling various OpenQuake GMPEs. If you want to modify specific
        behaviours for getting an estimate from a gmpe you should do it here.
        Modified from the old shakemap.shakelib.multigmpe.__get_mean_and_stddevs method.

        Args:
            sites:
            rup:
            dists:
            imt:
            stddev_types:

        Returns:

        """
        # Populate required site depth parameters for this gmpe in a new SitesContext
        sites = self.set_sites_depth_parameters(sites, gmpe)

        # Select the IMT
        gmpe_imts = [imt.__name__ for imt in
                     gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES]
        if (isinstance(imt, PGV)) and ("PGV" not in gmpe_imts):
            timt = SA(1.0)
        else:
            timt = imt

        # GMPE LIMITS come from the modules.conf file
        # e.g. self.GMPE_LIMITS <class 'dict'>: {'AbrahamsonEtAl2014RegJPN': {'vs30': ['150', '2000']}}
        # Apply any matching GMPE_LIMITS
        if self.GMPE_LIMITS:
            gmpes_with_limits = list(self.GMPE_LIMITS.keys())
            gmpe_class_str = str(gmpe).replace('()', '')
            if gmpe_class_str in gmpes_with_limits:
                limit_dict = self.GMPE_LIMITS[gmpe_class_str]
                # Note: The only gmp_limit item that is currently supported is vs30. This operation updates the
                # sites vs30 array, changing values which fall outside of the range defined in the conf file to
                # the edge value.   10 <= vs30 <= 100: [5 10 55 75 80 100 210] -> [10 10 55 75 80 100 100]
                for k, v in limit_dict.items():
                    if k == 'vs30':
                        vs30min = float(v[0])
                        vs30max = float(v[1])
                        sites.vs30 = np.clip(sites.vs30, vs30min, vs30max)
                    else:
                        logging.warn("GmpeRunner: GMPE_LIMITS key {0} is not yet implemented and will be "
                                     "ignored.".format(k))
        # Get the GMPE's mean and stdev
        lmean, lsd = gmpe.get_mean_and_stddevs(sites, rup, dists, timt,
                                               stddev_types)

        # Inflate the standard deviations to account for the point-source to finite rupture conversion.
        lsd_new = self._inflate_ps_sigma(gmpe, lmean, lsd, sites, rup,
                                      dists, timt, stddev_types)
        for sd in lsd:
            lsd_new.append(sd)
        lsd = lsd_new

        # If IMT is PGV and PGV is not given by the GMPE, then convert from PSA10.
        if (isinstance(imt, PGV)) and ("PGV" not in gmpe_imts):
            nh82 = NewmarkHall1982()
            lmean = nh82.convertAmps('PSA10', 'PGV', lmean)
            # Put the extra sigma from NH82 into intra event and total
            for j, stddev_type in enumerate(stddev_types):
                if stddev_type == const.StdDev.INTER_EVENT:
                    continue
                lsd[j] = nh82.convertSigmas('PSA10', 'PGV', lsd[j])

        if has_site is False:
            lamps = self.get_site_factors(
                sites, rup, dists, timt, default=True)
            lmean = lmean + lamps

        # Conversions due to component definition
        imc_in = gmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT
        imc_out = self.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT
        if imc_in != imc_out:
            bk17 = BooreKishida2017(imc_in, imc_out)
            lmean = bk17.convertAmps(imt, lmean, dists.rrup, rup.mag)
            #
            # The extra sigma from the component conversion appears to
            # apply to the total sigma, so the question arises as to
            # how to apportion it between the intra- and inter-event
            # sigma. Here we assume it all enters as intra-event sigma.
            #
            for j, stddev_type in enumerate(stddev_types):
                if stddev_type == const.StdDev.INTER_EVENT:
                    continue
                lsd[j] = bk17.convertSigmas(imt, lsd[j])

        # At this point lsd will have 2 * len(stddev_types) entries, the
        # first group will have the point-source to finite rupture
        # inflation (if any), and the second set will not; in cases where
        # a finite rupture is used, the two sets will be identical
        return lmean, lsd

    # TODO - implement this method. It should take the weights and results and combine them like the function
    #  currently does in multigmpe
    def _combine_gmpe_results(self, gmpe_results, large_dist=False):
        return (None,None)

    def compute_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types, output_branches=True):
        """
        :param args:
        :param merge_branches:
        :param kwargs:
        :return: if merge, then two arrays with dim (n_stations, len(self.gmpes) and
                    (n_stations,n_stdevs,len(self.gmpes))
                 else, two arrays with dim (n_stations, 1) and (n_stations, n_stdevs, 1)
        """
        # TODO - Some questions: What are the inputs to a gmpe in OQ and how does this preprocessing relate?

        gmpe_results = {}

        # Flatten data arrays in site and dists contexts prior to passing to the GMPEs
        old_shapes_sites = self.flatten_context_arrays(sites)
        old_shapes_dists = self.flatten_context_arrays(dists)

        unique_context_dims = set([x for x in old_shapes_sites.values()] + [x for x in old_shapes_dists.values()])
        if len(unique_context_dims) != 1:
            raise Exception(
                'All sites and dists elements must have same shape.')
        orig_shape = unique_context_dims.pop()

        # Note: This is a list of available sd types for each gmpe
        sd_avail = self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
        if not sd_avail.issuperset(set(stddev_types)):
            raise Exception("Requested an unavailable stddev_type.")

        # Evaluate GMPEs and apply any post-processing to their outputs
        for i, gmpe in enumerate(self.GMPES):
            gmpe_results[str(gmpe)] = self._oq_gmpe_get_mean_and_stddevs(gmpe, self.HAS_SITE[i], sites, rup, dists, imt,
                                                                         stddev_types)

        # TODO - Need to review the efficiency of how we are handling these arrays. Might make more sense to work
        #  with a mutable object (i.e. a list) so that we can apply these updates without having to unpack/repack the
        #  results each time we want to perform an operation on them.

        # Undo reshapes of inputs
        self.restore_context_array_shapes(sites, old_shapes_sites)
        self.restore_context_array_shapes(dists, old_shapes_dists)

        # Compute merged result
        combined = self._combine_gmpe_results(gmpe_results)

        # this applies squashed dist values to the combined value
        if self.CUTOFF_DISTANCE:
            # Re-combine the gmpe results with the long-distance weights
            lnmu_large, lnsd_large = self._combine_gmpe_results(gmpe_results, large_dist=True)
            lnmu, lnsd = combined
            # Stomp on lnmu and lnsd at large distances
            dist_cutoff = self.CUTOFF_DISTANCE
            lnmu[dists.rjb > dist_cutoff] = lnmu_large[dists.rjb > dist_cutoff]
            for i in range(len(lnsd)):
                lnsd[i][dists.rjb > dist_cutoff] = \
                    lnsd_large[i][dists.rjb > dist_cutoff]
            combined = (lnmu, lnsd)

        # Prepare the output dict
        output = {
            GmpeRunner.COMBINED_GMPE_KEY: combined
        }
        if output_branches:
            output = {**gmpe_results, **output}

        # Reshape all output arrays to match the un-flattened site dimensions
        for result in output:
            out_mean, out_sd = output[result]
            out_mean = np.reshape(out_mean, orig_shape)
            for i in range(len(out_sd)):
                out_sd[i] = np.reshape(out_sd[i], orig_shape)
            output[result] = (out_mean, out_sd)
        return output

    # TODO - This method needs to be updated
    def _inflate_ps_sigma(self, gmpe, lmean, lsd, sites, rup, dists, imt,
                       stddev_types):
        """
        If the point-source to finite-fault factors are used, we need to
        inflate the intra-event and total standard deviations. We do this
        by standard propagation of error techniques: taking the (numerical)
        derivative of the GMPE (as a function of distance) squared times the
        additional variance from the conversion, added
        to the variance of the GMPE (then taking the square root). We do
        this separately for each of Rrup and Rjb and sum the results.
        If Rrup and Rjb are calculated from a finite rupture model, their
        variance arrays will be "None" and lsd will remain unchanged.
        Otherwise the error inflation will be applied. Normally one or the
        other of Rrup/Rjb will not be used and so that term will be zero; in
        some cases both may be used and both may result in non-zero
        derivatives.

        Args:
            gmpe:
                The GMPE to use for the calculations. Must be a base GMPE and
                not a GMPE set, otherwise no action is taken.
            lmean:
                The mean values returned by the "normal" evaluation of the
                GMPE.
            lsd:
                The standard deviations returned by the "normal" evaluation
                of the GMPE.
            sites:
                The sites context required by the GMPE.
            rup:
                The rupture context required by the GMPE.
            dists:
                The distance context required by the GMPE.
            imt:
                The intensity measure type being evaluated.
            stddev_types:
                The list of stddev types found in lsd.

        Returns:
            list: A list of arrays of inflated standard deviations
            corresponding to the elements of lsd.
        """
        new_sd = []
        delta_distance = 0.01
        delta_var = [0, 0]
        for i, dtype in enumerate(('rrup', 'rjb')):
            # Skip dtype if the gmpe does not require it
            if dtype not in gmpe.REQUIRES_DISTANCES:
                continue
            # Skip dtype if it has not been subject to a point-source to
            # finite rupture conversion
            dvar = getattr(dists, dtype + '_var', None)
            if dvar is None:
                continue
            # Add a small amound to the rupture distance (rrup or rjb)
            # and re-evaluate the GMPE
            rup_dist = getattr(dists, dtype)
            rup_dist += delta_distance
            tmean, tsd = gmpe.get_mean_and_stddevs(sites, rup, dists, imt,
                                                   stddev_types)
            # Find the derivative w.r.t. the rupture distance
            dm_dr = (lmean - tmean) / delta_distance
            # The additional variance is (dm/dr)^2 * dvar
            delta_var[i] = dm_dr ** 2 * dvar
            # Put the rupture distance back to what it was
            rup_dist -= delta_distance
        for i, stdtype in enumerate(stddev_types):
            if stdtype == const.StdDev.INTER_EVENT:
                new_sd.append(lsd[i].copy())
                continue
            new_sd.append(np.sqrt(lsd[i] ** 2 + delta_var[0] + delta_var[1]))
        return new_sd

    # TODO - This function needs to be updated
    def get_site_factors(self, sites, rup, dists, imt, default=False):
        """
        Method for computing site amplification factors from the defalut GMPE
        to be applied to GMPEs which do not have a site term.

        **NOTE** Amps are calculated in natural log units and so the ln(amp)
        is returned.

        Args:
            sites (SitesContext): Instance of SitesContext.
            rup (RuptureContext): Instance of RuptureContext.
            dists (DistancesContext): Instance of DistancesContext.
            imt: An instance openquake.hazardlib.imt.
            default (bool): Boolean of whether or not to return the
                amplificaiton factors for the gmpes or default_gmpes_for_site.
                This argument is primarily only intended to be used internally
                for when we just need to access the default amplifications to
                apply to those GMPEs that do not have site terms.

        Returns:
            Site amplifications in natural log units.
        """

        # ---------------------------------------------------------------------
        # Make reference sites context
        # ---------------------------------------------------------------------

        ref_sites = copy.deepcopy(sites)
        ref_sites.vs30 = np.full_like(sites.vs30, self.REFERENCE_VS30)

        # ---------------------------------------------------------------------
        # If default True, construct new MultiGMPE with default GMPE/weights
        # ---------------------------------------------------------------------
        if default is True:
            tmp = MultiGMPE.from_list(
                self.DEFAULT_GMPES_FOR_SITE,
                self.DEFAULT_GMPES_FOR_SITE_WEIGHTS,
                self.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT)

        # ---------------------------------------------------------------------
        # If default False, just use self
        # ---------------------------------------------------------------------
        else:
            tmp = self

        lmean, lsd = tmp.get_mean_and_stddevs(
            sites, rup, dists, imt,
            list(tmp.DEFINED_FOR_STANDARD_DEVIATION_TYPES))
        lmean_ref, lsd = tmp.get_mean_and_stddevs(
            ref_sites, rup, dists, imt,
            list(tmp.DEFINED_FOR_STANDARD_DEVIATION_TYPES))

        lamps = lmean - lmean_ref

        return lamps

    @classmethod
    def from_config(cls, conf, filter_imt=None):
        """
        Construct a GmpeRunner from a config dict.

        Args:
            conf (dict): Dictionary of config options.
            filter_imt (IMT): An optional IMT to filter/reweight the GMPE list.

        Returns:
            MultiGMPE object.

        """
        IMC = getattr(const.IMC, conf['interp']['component'])
        selected_gmpe = conf['modeling']['gmpe']

        logging.debug('selected_gmpe: %s' % selected_gmpe)
        logging.debug('IMC: %s' % IMC)

        # ---------------------------------------------------------------------
        # Allow for selected_gmpe to be found in either conf['gmpe_sets'] or
        # conf['gmpe_modules'], if it is a GMPE set, then all entries must be
        # either a GMPE or a GMPE set (cannot have a GMPE set that is a mix of
        # GMPEs and GMPE sets).
        # ---------------------------------------------------------------------

        if selected_gmpe in conf['gmpe_sets'].keys():
            selected_gmpe_sets = conf['gmpe_sets'][selected_gmpe]['gmpes']
            gmpe_set_weights = \
                [float(w) for w in conf['gmpe_sets'][selected_gmpe]['weights']]
            logging.debug('selected_gmpe_sets: %s' % selected_gmpe_sets)
            logging.debug('gmpe_set_weights: %s' % gmpe_set_weights)

            # -----------------------------------------------------------------
            # If it is a GMPE set, does it contain GMPEs or GMPE sets?
            # -----------------------------------------------------------------

            set_of_gmpes = all([s in conf['gmpe_modules'] for s in
                                selected_gmpe_sets])
            set_of_sets = all([s in conf['gmpe_sets'] for s in
                               selected_gmpe_sets])

            if set_of_sets is True:
                mgmpes = []
                for s in selected_gmpe_sets:
                    mgmpes.append(cls.__multigmpe_from_gmpe_set(
                        conf, s, filter_imt=filter_imt))
                out = GmpeRunner.from_list(mgmpes, gmpe_set_weights, imc=IMC)
            elif set_of_gmpes is True:
                out = cls.__multigmpe_from_gmpe_set(
                    conf,
                    selected_gmpe,
                    filter_imt=filter_imt)
            else:
                raise TypeError("%s must consist exclusively of keys in "
                                "conf['gmpe_modules'] or conf['gmpe_sets']"
                                % selected_gmpe)
        elif selected_gmpe in conf['gmpe_modules'].keys():
            modinfo = conf['gmpe_modules'][selected_gmpe]
            mod = import_module(modinfo[1])
            tmpclass = getattr(mod, modinfo[0])
            out = GmpeRunner.from_list([tmpclass()], [1.0], imc=IMC)
        else:
            raise TypeError("conf['modeling']['gmpe'] must be a key in "
                            "conf['gmpe_modules'] or conf['gmpe_sets']")

        out.DESCRIPTION = selected_gmpe

        # ---------------------------------------------------------------------
        # Deal with GMPE limits
        # ---------------------------------------------------------------------
        gmpe_lims = conf['gmpe_limits']

        # We need to replace the short name in the dictionary key with module
        # name here since the conf is not available within the MultiGMPE class.
        mods = conf['gmpe_modules']
        mod_keys = mods.keys()
        for k, v in gmpe_lims.items():
            if k in mod_keys:
                gmpe_lims[mods[k][0]] = gmpe_lims.pop(k)

        out.GMPE_LIMITS = gmpe_lims

        return out

    def __multigmpe_from_gmpe_set(conf, set_name, filter_imt=None):
        """
        Private method for constructing a GmpeRunner from a set_name.

        Args:
            conf (ConfigObj): A ShakeMap config object.
            filter_imt (IMT): An optional IMT to filter/reweight the GMPE list.
            set_name (str): Set name; must correspond to a key in
                conf['set_name'].

        Returns:
            GmpeRunner

        """
        IMC = getattr(const.IMC, conf['interp']['component'])

        selected_gmpes = conf['gmpe_sets'][set_name]['gmpes']
        selected_gmpe_weights = \
            [float(w) for w in conf['gmpe_sets'][set_name]['weights']]

        # Check for large distance GMPEs
        if 'weights_large_dist' in conf['gmpe_sets'][set_name].keys():
            if not conf['gmpe_sets'][set_name]['weights_large_dist']:
                selected_weights_large_dist = None
            else:
                selected_weights_large_dist = \
                    [float(w) for w in
                     conf['gmpe_sets'][set_name]['weights_large_dist']]
        else:
            selected_weights_large_dist = None

        if 'dist_cutoff' in conf['gmpe_sets'][set_name].keys():
            if np.isnan(conf['gmpe_sets'][set_name]['dist_cutoff']):
                selected_dist_cutoff = None
            else:
                selected_dist_cutoff = \
                    float(conf['gmpe_sets'][set_name]['dist_cutoff'])
        else:
            selected_dist_cutoff = None

        if 'site_gmpes' in conf['gmpe_sets'][set_name].keys():
            if not conf['gmpe_sets'][set_name]['site_gmpes']:
                selected_site_gmpes = None
            else:
                selected_site_gmpes = \
                    conf['gmpe_sets'][set_name]['site_gmpes']
        else:
            selected_site_gmpes = None

        if 'weights_site_gmpes' in conf['gmpe_sets'][set_name].keys():
            if not conf['gmpe_sets'][set_name]['weights_site_gmpes']:
                selected_weights_site_gmpes = None
            else:
                selected_weights_site_gmpes = \
                    conf['gmpe_sets'][set_name]['weights_site_gmpes']
        else:
            selected_weights_site_gmpes = None

        # ---------------------------------------------------------------------
        # Import GMPE modules and initialize classes into list
        # ---------------------------------------------------------------------
        gmpes = []
        for g in selected_gmpes:
            mod = import_module(conf['gmpe_modules'][g][1])
            tmpclass = getattr(mod, conf['gmpe_modules'][g][0])
            gmpes.append(tmpclass())

        # ---------------------------------------------------------------------
        # Filter out GMPEs not applicable to this period
        # ---------------------------------------------------------------------
        if filter_imt is not None:
            filtered_gmpes, filtered_wts = filter_gmpe_list(
                gmpes, selected_gmpe_weights, filter_imt)
        else:
            filtered_gmpes, filtered_wts = gmpes, selected_gmpe_weights

        # ---------------------------------------------------------------------
        # Import site GMPEs
        # ---------------------------------------------------------------------
        if selected_site_gmpes is not None:
            if isinstance(selected_site_gmpes, str):
                selected_site_gmpes = [selected_site_gmpes]
            site_gmpes = []
            for g in selected_site_gmpes:
                mod = import_module(conf['gmpe_modules'][g][1])
                tmpclass = getattr(mod, conf['gmpe_modules'][g][0])
                site_gmpes.append(tmpclass())
        else:
            site_gmpes = None

        # ---------------------------------------------------------------------
        # Filter out site GMPEs not applicable to this period
        # ---------------------------------------------------------------------
        if site_gmpes is not None:
            if filter_imt is not None:
                filtered_site_gmpes, filtered_site_wts = filter_gmpe_list(
                    site_gmpes, selected_weights_site_gmpes, filter_imt)
            else:
                filtered_site_gmpes = copy.copy(site_gmpes)
                filtered_site_wts = copy.copy(selected_weights_site_gmpes)
        else:
            filtered_site_gmpes = None
            filtered_site_wts = None

        # ---------------------------------------------------------------------
        # Construct MultiGMPE
        # ---------------------------------------------------------------------
        logging.debug('    filtered_gmpes: %s' % filtered_gmpes)
        logging.debug('    filtered_wts: %s' % filtered_wts)

        mgmpe = GmpeRunner.from_list(
            filtered_gmpes, filtered_wts,
            default_gmpes_for_site=filtered_site_gmpes,
            default_gmpes_for_site_weights=filtered_site_wts,
            imc=IMC)

        # ---------------------------------------------------------------------
        # Append large-distance info if specified
        # ---------------------------------------------------------------------
        if selected_dist_cutoff is not None:
            if filter_imt is not None:
                filtered_gmpes_ld, filtered_wts_ld = filter_gmpe_list(
                    gmpes, selected_weights_large_dist, filter_imt)
            else:
                filtered_wts_ld = copy.copy(selected_weights_large_dist)

            mgmpe.CUTOFF_DISTANCE = copy.copy(selected_dist_cutoff)
            mgmpe.WEIGHTS_LARGE_DISTANCE = copy.copy(filtered_wts_ld)

        mgmpe.DESCRIPTION = set_name
        return mgmpe

    @staticmethod
    def set_sites_depth_parameters(input_sites, gmpe):
        """
        Need to select the appropriate z1pt0 value for different GMPEs.
        Note that these are required site parameters, so even though
        OQ has these equations built into the class in most cases.
        I have submitted an issue to OQ requesting subclasses of these
        methods that do not require the depth parameters in the
        SitesContext to make this easier.

        Args:
            input_sites:1 An OQ sites context.
            gmpe: An OQ GMPE instance.

        Returns:
            An NEW OQ sites context with the depth parameters set for the
            requested GMPE.

        """
        # Make a copy of sites if you want to actually return a new sites context
        sites = copy.deepcopy(input_sites)

        if gmpe == 'MultiGMPE()':
            return sites

        sites = Sites._addDepthParameters(sites)

        if gmpe == 'AbrahamsonEtAl2014()' or \
           gmpe == 'AbrahamsonEtAl2014RegTWN()' or \
           gmpe == 'AbrahamsonEtAl2014RegCHN()':
            sites.z1pt0 = sites.z1pt0_ask14_cal
        if gmpe == 'AbrahamsonEtAl2014RegJPN()':
            sites.z1pt0 = sites.z1pt0_ask14_jpn
        if gmpe == 'ChiouYoungs2014()' or \
           isinstance(gmpe, BooreEtAl2014):
            sites.z1pt0 = sites.z1pt0_cy14_cal
        if isinstance(gmpe, CampbellBozorgnia2014):
            if gmpe == 'CampbellBozorgnia2014JapanSite()' or \
               gmpe == 'CampbellBozorgnia2014HighQJapanSite()' or \
               gmpe == 'CampbellBozorgnia2014LowQJapanSite()':
                sites.z2pt5 = sites.z2pt5_cb14_jpn
            else:
                sites.z2pt5 = sites.z2pt5_cb14_cal
        if gmpe == 'ChiouYoungs2008()' or \
           gmpe == 'Bradley2013()' or \
           gmpe == 'Bradley2013Volc()':
            sites.z1pt0 = sites.z1pt0_cy08
        if gmpe == 'CampbellBozorgnia2008()':
            sites.z2pt5 = sites.z2pt5_cb07
        if gmpe == 'AbrahamsonSilva2008()':
            sites.z1pt0 = gmpe._compute_median_z1pt0(sites.vs30)
        return sites

    def describe(self):
        """
        Construct a dictionary that describes the MultiGMPE.

        Note: For simplicity, this method ignores issues related to
        GMPEs used for the site term and changes in the GMPE with
        distance. For this level of detail, please see the config files.

        Returns:
            A dictionary representation of the MultiGMPE.
        """
        gmpe_dict = {
            'gmpes': [],
            'weights': [],
            'name': self.DESCRIPTION
        }

        for i in range(len(self.GMPES)):
            gmpe_dict['weights'].append(self.WEIGHTS[i])
            if isinstance(self.GMPES[i], MultiGMPE):
                gmpe_dict['gmpes'].append(
                    self.GMPES[i].describe()
                )
            else:
                gmpe_dict['gmpes'].append(str(self.GMPES[i]))

        return gmpe_dict

    def filter_gmpe_list(gmpes, wts, imt):
        """
        Method to remove GMPEs from the GMPE list that are not applicable
        to a specific IMT. Rescales the weights to sum to one.

        Args:
            gmpes (list): List of GMPE instances.
            wts (list): List of floats indicating the weight of the GMPEs.
            imt (IMT): OQ IMT to filter GMPE list for.

        Returns:
            tuple: List of GMPE instances and list of weights.

        """
        if wts is None:
            n = len(gmpes)
            wts = [1 / n] * n

        per_max = [np.max(get_gmpe_sa_periods(g)) for g in gmpes]
        per_min = [np.min(get_gmpe_sa_periods(g)) for g in gmpes]
        if imt == PGA():
            sgmpe = [g for g in gmpes if imt in
                     get_gmpe_coef_table(g).non_sa_coeffs]
            swts = [w for g, w in zip(gmpes, wts) if imt in
                    get_gmpe_coef_table(g).non_sa_coeffs]
        elif imt == PGV():
            sgmpe = []
            swts = []
            for i in range(len(gmpes)):
                if (imt in get_gmpe_coef_table(gmpes[i]).non_sa_coeffs) or \
                        (per_max[i] >= 1.0 and per_min[i] <= 1.0):
                    sgmpe.append(gmpes[i])
                    swts.append(wts[i])
        else:
            per = imt.period
            sgmpe = []
            swts = []
            for i in range(len(gmpes)):
                if (per_max[i] >= per and per_min[i] <= per):
                    sgmpe.append(gmpes[i])
                    swts.append(wts[i])

        if len(sgmpe) == 0:
            raise KeyError('No applicable GMPEs from GMPE list for %s' % str(imt))

        # Scale weights to sum to one
        swts = np.array(swts)
        swts = swts / np.sum(swts)

        return sgmpe, swts

