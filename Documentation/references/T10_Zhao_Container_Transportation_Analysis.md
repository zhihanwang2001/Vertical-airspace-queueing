# T10Literature Analysis: averagebalancecardvehicleschedulingstrategyforrequiresrequestouterincludecodeheadbetweensetinstallboxoperatetransport

**Full Citation**: Zhao, Y., Ji, Y., & Zheng, Y. (2025). "Balanced Truck Dispatching Strategy for Inter-Terminal Container Transportation with Demand Outsourcing." *Mathematics*, 13(2), 287. DOI: 10.3390/math13020287.

---

## 📄 Paper Basic Information (selfmoveproposetake)

* Title: Balanced Truck Dispatching Strategy for Inter-Terminal Container Transportation with Demand Outsourcing
* Authors: Yucheng Zhao, Yuxiong Ji, Yujing Zheng
* Publication Venue: MDPI Mathematics
* Year: 2025
* Theory Type: Comprehensive Modeling (closedequation Jackson network + costminimizeoptimization + MVA solutionanalysis + evolveizationalgorithm)

# 🔬 coretheoryframeworkunitsscoreanalysis

1. **queueingsystemtypetype**

* standardmodel: **Closed Jackson network** (closedequation Jackson network; cardvehicleiscustomer, endend/pathsegmentisservicesectionpoint). seeFig2 (p.6)and §3.1–3.2. 
* toreachprocess: **Poisson** (taskalivebecomeservicefromPoisson; §3.1"Task generation"needpoint, p.4–5). 
* serviceprocess: **onegeneral/certainfixedparameternumberindicatenumberizationTabledescription**

 * endendsectionpoint: singleserviceplatform, servicerate μ= d_i (etc.valueinindicatenumberserviceassumption); 
 * pathbysectionpoint: multipleserviceplatform c_i=a_r, servicerate μ=1/b_r (byrowprocesswhenbetweendecidefixed). seeequation(1)(2) (p.5–6). 
* systemcapacity: **finitecapacity/congestionpenalty**

 * endendcapacity h_i, pathsegment"servicedevicenumber" a_r; andobjectivefunctionnumbercontaincongestionpenaltyitemandcapacityconstraint (equation(5)(6), p.6–7, Table1 p.5). 
* systemstructure: **networkqueueing** (endendandpathsegmenttwotypesectionpointconstructbecomenetwork; Fig2 p.6). 

2. **scorelayer/verticalstructure**

* layerlevelfixedmeaning: no"objectmanageverticallayer"fixedmeaning; onlyhave**logicontwotypesectionpoint** (endendlayer vs pathbylayer). seeFig2 (p.6). 
* layerbetweenrelationship: **related/transfer** (cardvehicleinendendandpathbysectionpointbetweentransfer, transfergeneralrate γ_ij by(r_ij − s_ij)induceguide; equation(3)(4)andFig2annotation, p.6). 
* capacityallocationplacement: **nonmeanuniform, staticstategivefixed** (h_i, a_r nonmeanuniform; equation(6)andTable1, p.5–7). 

3. **systemmovestatemechanism**

* movestatetransfer: **have** (vehiclevehicleinnetworkintransferby γ_ij decidefixed, γ receivemainmoveouterinclude s_ij adjust, seeFig2and §3.1–3.2, p.5–6). 
* statedependency: **partscoreexistin**

 * servicerate μ fixedfixed; 
 * but"reverseshouldequationouterinclude"sendalivein**nocanusesvehicle**when (statedrivemove), andcongestionpenaltyfollowinvenuevehiclevehiclecount variation (equation(5), p.6–7; §3.1, p.4–5). 
* load balancing: **based onouterincluderatiostaticstate—standardmovestatescheduling** (s_ij mainmoveouterincluderatiorateasdecisionchangequantity; andvehicleteamscale F connectcombineoptimization. §3.2–3.4; Table6providesmostsuperior s_ij/ r_ij, p.11). 

# 🔍 and MCRPS/D/K theoryprecisecertainComparison

**ourtheory (MCRPS/D/K)needpointreturncustomer**: multiplelayerrelatedtoreach, randombatchquantityservice, Poissonscoreflow, statedependency, pressuretriggermovestatetransfer, finitecapacity, vertical5layerandinverted pyramidcapacityetc.. 

| dimensionaldegree | discussionpaperdomethod | and MCRPS/D/K relationship |
| ------------------------------------ | ------------------------------------- | ------------------------------- |
| Multi-layer correlated arrivals (MC) | tasktoreachisPoisson, each OD ratio r_ij fixedfixed; notshowequationconstruct"multiplelayerrelatedtoreach" | **mismatch** (nomultiplelayerrelatedtoreach) |
| Random batch service (R) | nobatchquantityservice; endendsingleserviceplatform, pathsegmentmultipleserviceplatformbutnonbatchservice | **mismatch** |
| Poisson splitting (P) | r_ij and s_ij canviewworkforPoissonflow**ratioscoreflow/sparsesparseization** (outerinclude) | **partscorephasesimilar** (conceptonhavePoissonscoreflow/sparseexplain) |
| State-dependent (S) | reverseshouldequationouterincludereceive"whetherhaveemptyvehicle"impact; congestionpenaltyfollowinvenuenumberchange | **partscorematchallocation** (substitutevalue/outerincludepresentstatedependency; serviceratebookbodynotfollowstatechange) |
| Dynamic transfer (D) | vehiclevehicleinnetworkinaccording γ_ij transfer; no"pressuretrigger"**strategypropertytransfer**mechanism | **mismatch** (nopressuretriggertransferlogic) |
| Finite capacity (K) | endendcapacity h_i, pathsegmentservicedevicenumber a_r andtotalbody F constraint | **matchallocation** |
| verticalscorelayer (5layerhighempty) | **no**verticalspacescorelayer, onlylogictwolayersectionpoint | **mismatch** |
| inverted pyramidcapacity {8,6,4,3,2} | **no**, capacityisvenuestationgivefixed | **mismatch** |

> keyposition: Poisson assumptionandclosednetworkstructure (§3.1, p.4–5), γ_ij by (r_ij − s_ij) guideexit (Fig2, p.6), Optimization Objectivecontainouterincludeandcongestionpenalty (equation(5), p.6–7), MVA passpush (Table2, p.8), resultsand s_ij ratio (Table6, p.11). 

# 🧪 theoryinnovationpropertyverification (1–10score)

1. whetherexistin**completeallphasesame** MCRPS/D/K system？**1/10**

 * this paperisclosedequation Jackson + outerincludeoptimization; nomultiplelayerrelatedtoreach, nobatchquantityservice, noverticallayerandinverted pyramid, nopressuretriggertransfer. 
2. whetherhave**verticalspacescorelayer**queueingmodeling？**0/10** (notinvolveand). 
3. whetherhave**inverted pyramidcapacityallocationplacement**theory？**0/10** (notinvolveand). 
4. whetherhave**relatedtoreach+batchquantityservice+Poissonscoreflow**combination？**2/10**

 * onlycantreat (r_ij, s_ij) viewwork**Poissonflowratioscoreflow/sparseexplain**; notseerelatedtoreachandbatchservice. 
5. whetherhave**pressuretriggermovestatetransfer**mechanism？**1/10**

 * outerincludehave"emptyvehiclestatetrigger"reverseshouldequationlogic; butno"layerbetweenpressuretriggertransfer". 

**verificationresults**

* ✅ **completealloriginal** (targetforourtheorymainsheet): 

 * our**verticalscorelayer (5layerhighempty)+ inverted pyramidcapacity + pressuretriggerundertowardtransfer + multi-objectiveGinireward + hybridaction (continuous/discrete)** MCRPS/D/K frameworkunits, inthis paperin**meannotexitappear**, thereforeThesecoresettingComparisonthis paperis**actualqualitypropertyoriginal**. 
* ⚠️ **partscorephasesimilar**: 

 * onlyin"**Poissonscoreflow/sparseexplain**"idea (r_ij and s_ij)and"**statetriggerouterinclude** (nocanusesvehiclewhen)"onexistin**finitephasesimilar**; butthisbelongin**openreleaserequiresrequest—closednetworkvehiclevehicle**couplecombineunderschedulingfinesection, notinvolveandourstrongadjustverticallayerlevel/batchservice/pressuretransfer. 
* 🔄 **canreferencetheory**: 

 * **closedequation Jackson network + MVA** (Table2 p.8)canasourverticalscorelayersystemin"vehiclevehicle (ornopersonmachine)followloopresource"viewjiaounder**cancomputeapproximate**and**performanceevaluates**worktool; forournotcome**halfopenequation/hybridnetwork**scoreanalysisalsohaveenablesend. 
* ❌ **existinconflict**: 

 * nodirecttheoryconflict; modelsidefocus ondifferent (this paperoptimizationouterincludeandvehicleteamscale, nonverticalairspacelayerleveldesign). 

# 💡 forourtheoryvaluevalue

1. **theoryfoundationsupport**

 * this paperuses**closedequation Jackson + MVA**provides**faststablehealthy**performancecomputeandgeneralratequantity (π_i(t,Q))estimateplanframeworkunits (§3.3.1, Table2), canprovideourinverticalscorelayer UAV systemin, **withvehiclevehicle/nopersonmachineis"followloopcustomer"**for**queuestablestatequantity**approximate; canandour**multi-objectivereward** (containGinifairness)parallelschoolstandard. 
2. **poordifferenceizationverification**

 * inrelatedworkworkreviewandmethodComparisonin, clearcertainindicateexit: existingclosednetwork/outerincludeoptimizationstudyresearch**notintroducingverticalspacelayer, inverted pyramidcapacity, pressuretriggertransferandrandombatchquantityservice**; ourcanusesthis paperas"**network-outerincluderangeequation**"substituteTablepapercontribute, convexshowourfrom**spacelayerlevelandobjectmanagemechanism**exitsendpoordifferenceization. 
3. **numberlearningworktoolreference**

 * reference: **(i)** MVA passpush (Table2 p.8); **(ii)** closednetwork**flowguardconstantandreturnoneization** (equation(3)(4) p.6; equation(10)–(13) p.7–8); **(iii)** inourmodelintreat**layerbetweentransfergeneralrate**writebecome γ_ℓℓ′(a), orderitsby**pressure/congestionstate**and**controlaction**commonsamedecidefixed, useswithpushwidethis paper γ by (r−s) decidefixedapproach. 
4. **citeusesstrategy**

 * in**Related Work (queueingnetworkandresourcefollowloop)**partscoreciteusesthis paper, fixedpositionis"**closedequation Jackson + outerincluderatiorateoptimization**"substituteTable; in**methoddiscussion**inwith"**performanceevaluatesbaseline**"citeusesits **MVA** implementation; in**Discusses/notcomeworkwork**inforaccordingits"suggestionTreatsclosednetworkextensionishalfopenreleasenetwork"expandlook (§5 p.13), callshouldour**multiplelayerrelatedtoreach**changeonegeneralframeworkunits. 

# resultdiscussionarea (aspecttowardinvestdraftdiscussionproof)

* **theoryinnovationdegreecertainrecognize**: **9/10** (based onthispaperverification)

 * andthis paperphaseratio, our MCRPS/D/K in**layerlevelstructure (vertical5layer)**, **capacityformstate (inverted pyramid)**, **toreachrelatedproperty**, **randombatchquantityservice**and**pressuretriggercrosslayertransfer**etc.keystructurepropertyassumptionon**meanisnewincrease**and**changestrong**theorysetting; this papercanas**differentrangeequation**network-outerincludetypeforaccording. 
* **ourinnovationuniqueproperty**: **completeallunique** (in"verticalscorelayer+inverted pyramid+pressuretriggertransfer+batchquantity/statedependency+hybridaction"**combinationbody**meaningmeaningon). 

---

**supplementfill: discussionpaperinkeyactualproofpoint**

* planexample: onseaoceanmountainportfourendend, provides OD requiresrequestandrowprocesswhenbetween/cost (Table4, Table5, p.10–11). 
* optimizationresults: connectcombineoptimizationvehicleteamscaleandmainmoveouterincluderate, **totalcostcomparepurereverseshouldouterincludefall 9.8%** (Table6, p.11; Fig5cost—vehicleteamscalecurves p.12). 
* methodchain: **DE yuanenablesendequation**requestsolution (§3.3.2, p.8–9)+ **MVA**evaluates (Table2, p.8). 

---

**theoryinnovationrelateddegree**: **low** (networkqueueingfoundationphasenear, spacescorelayer/batchquantity/scoreflowmechanismcompletealldifferent)
**ourinnovationuniquepropertycertainrecognize**: **completeallunique**
**suggestionadjuststudyprioritizedlevel**: **inetc.** (mainlyasclosedequationnetworkperformanceevaluatesworktoolreference)

---

**Analysis Completion Date**: 2025-01-28 
**Analysis Quality**: Detailed analysis withclosedequationJacksonnetworkmodelingandMVAsolutionanalysismethod 
**Recommended Use**: asclosedequationnetworkoptimizationmethodreference, referenceMVAsolutionanalysistechniqueandouterincludestrategyoptimizationidea