#define DRIVER_VERSION "6"

/* CHANGE LOG
 * 6: Fix corner case bug
 * 5: Final score normalization & Updating Eqfreq
 * 4: Updating Eqfreq calculation
 * 3: Updating input file parser
 * 2: Adding branch length functionality
 * 1: Modified the logic in parsing FASTA names 
 */

#include "caster-site-workflow.hpp"

int GENE_ID = 0;

int main(int argc, char** argv){
    configureCasterSiteArguments();
    
    #ifdef CUSTOMIZED_ANNOTATION_TERMINAL_LENGTH
    cerr << "Warning: This version can be much slower and require much more memory than the regular version. You might want to compute the correct topology first and add branch lengths with this version on fix topology.\n";
    #endif

    Workflow WF(argc, argv);
	ARG.getStringArg("annotation") = "BootstrapSupport";

    LOG << "#Base: " << WF.meta.tripInit.seq.len() << endl;
    auto res = WF.meta.run();
    LOG << "Normalized score: " << (double) res.first / 4 << endl;
	return 0;
}
