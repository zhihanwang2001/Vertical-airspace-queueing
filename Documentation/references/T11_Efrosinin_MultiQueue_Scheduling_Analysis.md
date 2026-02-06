# T11Literature Analysis: multiplequeuesysteminresultcombinesimulationandspiritthroughnetworkmostsuperiorscheduling

**Full Citation**: Efrosinin, D., Vishnevsky, V., & Stepanova, N. (2023). "Optimal Scheduling in General Multi-Queue System by Combining Simulation and Neural Network Techniques." *Sensors*, 23(19), 8055. DOI: 10.3390/s23198055.

---

## 📄 Paper Basic Information

* **Title**: Optimal Scheduling in General Multi-Queue System by Combining Simulation and Neural Network Techniques
* **Authors**: Dmitry Efrosinin, Vladimir Vishnevsky, Natalia Stepanova
* **Publication Venue**: MDPI *Sensors* (openreleaseobtaintake)
* **Year**: 2023
* **Theory Type**: **Comprehensive Modeling** (GI/G/1 parallelqueue + Markovcanhusbanddecisionprocess/strategyiteratesubstitute + eventdrivemovesimulation + spiritthroughnetwork + modelsimulateretreatfire), seeabstractandsection1section; systemstructureshowmeaningsee**Fig1 (p.5)**. 

---

# 🔬 coretheoryframeworkunitsscoreanalysis (⭐⭐⭐⭐⭐)

## 1) queueingsystemtypetype

* **standardmodel**: singleserviceplatformin**multipleparalleldifferencequalityqueue**ofbetweenroundturnreceivecontrolsystem, canviewis**polling type** (butadopting"queueclearemptywhendecision"*exhaustive*servicerules). onegeneralsituationformis **GI/G/1‖N parallel**; indicatenumbersituationformmappingis**continuouswhenbetween MDP**. modelandstatein§2–§3Formalizes, Fig1providesstructureshowmeaning. 

 * **toreachprocess**: eachqueue**onegeneralscoredistribution GI** (providestoreachbetweenseparatescoredistribution (A_i(t))), inverification/foraccordingwhenuses**indicatenumberscoredistribution**specialexample; sensitivefeelpropertyexperimentsreturncovercover Gamma/Lognormal/Pareto etc.. see§2 (p.5)and§7 (p.21–24). 
 * **serviceprocess**: eachqueue**onegeneralscoredistribution G** (providesservicewhenbetweenscoredistribution (B_i(t))); indicatenumberspecialexamplefor MDP andstrategyiteratesubstituteforstandard. see§2–§3. 
 * **systemcapacity**: MDPscoreanalysisrequires**truncatebreakslowrusharea (B_i<\infty)**; paperinshowexampleprovidestruncatebreak andexplainloadsetplacementwithfalllowloselosegeneralrate (p.8–10). simulationinthroughenoughlargeslowrushandwarm up/statisticswindowportimplementationaveragecostestimateplan. 
 * **systemstructure**: **parallelnetwork/singleservicepersonroundturn**; decisionsendalivein**certainonequeueclearemptydecisionhistoryyuan**. movestateplanningcalculatesub, Bellman methodprocessand**algorithm1 (strategyiteratesubstitute, p.9)**provides. 

## 2) scorelayer/verticalstructure

* **whetherscorelayer**: **notinvolveandspace/verticallayer**; onlyexistin"passiveservicequeue/itsremainderqueue"**logictwotyperoles**. discussionpapernohave"multiplelayer (L1…Lk)"or"onunderlayertransfer"modeling. Fig1 (p.5)isplaneparallelstructure. 
* **layerbetweenrelationship/capacityallocationplacement**: notsuitableuses (nolayer); eachqueue**nonmeanuniformparameternumber** (toreach/service/feeuses/switchchangesubstitutevalueetc.), capacitytruncatebreakin MDP ingivefixedwhilenonoptimization. 

## 3) systemmovestatemechanism

* **movestatetransfer**: have (**servicedeviceinqueuebetweenswitchchange**), butonlyin**servicecompleteandwhenfirstqueueclearempty**whensendalive, receive**switchchangesubstitutevalue**impact; decisionstrategycanis LQF, cμ or**spiritthroughnetworkoutput**; see§2–§5, **Fig3 (spiritthroughNetwork Architecture, p.13)**and**algorithm2 (eventdrivemovesimulation, p.12)**. 
* **statedependency**: **costandactionselectionforstatedependency** (queueinglength, whenfirstservicequeue); **servicerate/toreachratebookbodynotfollowstatechangeization** (ingivefixedscoredistributionunderfixedfixed). MDP i.e.whencostcontain**holdhavecost + switchchangecost** (§3, p.7). 
* **load balancing**: belongin**movestate/intelligentscheduling**problem; through**NN+modelsimulateretreatfire**directminimizeaveragecost, andand**strategyiteratesubstitute**under Markov specialexampleforstandard; **Fig4–5 (p.20)**showsreceiveconverge/costComparison. 

---

# 🔍 and MCRPS/D/K theoryprecisecertainComparison

> our MCRPS/D/K (multiplelayerrelatedtoreach, randombatchquantityservice, Poissonscoreflow, statedependency, pressuretriggermovestatecrosslayertransfer, finitecapacity; 5layervertical + inverted pyramidcapacity)is**verticalscorelayer+hybridmechanism**combinationbody. 

| dimensionaldegree | this paperdomethod | and MCRPS/D/K relationship |
| ------------- | ---------------------------------------- | --------------- |
| **MC** multiplelayerrelatedtoreach | queueindependent GI toreach; papercontributereviewproposeand**relatedtoreach**studyresearch, butthis papercoremodelnotintroducing | **mismatch** |
| **R** randombatchquantityservice | **nobatchquantity**, singleguestuserservice, queueclearemptyequation (exhaustive) | **mismatch** |
| **P** Poissonscoreflow | **notshowequationmodelingscoreflow**; onlyinindicatenumbersituationformunderis Poisson toreach, butno"scoreflow"structure | **mismatch/weakrelated** |
| **S** statedependency | decision/costforstatedependency; toreach/servicerateforstate**notdependency** | **partscorematchallocation** |
| **D** movestatetransfer | **haveservicedeviceswitchchange** (based onstate/substitutevalue), butnon"pressuretrigger**layerbetween**transfer" | **partscorematchallocation (mechanismdifferent)** |
| **K** finitecapacity | MDP requires**finiteslowrushtruncatebreak**; simulationinetc.valueprocessing | **matchallocation** |
| **verticalscorelayer** (5layer) | **no**spaceverticallayerandlayerbetweendynamics | **mismatch** |
| **inverted pyramidcapacity** | **no**thistypecapacityallocationplacementoroptimization | **mismatch** |

Evidence: systemandMDPformequation (§2–§3, Fig1, algorithm1), simulationandNN/SA (§4–§6, Fig3, algorithm2/4), numbervalueandsensitivefeelproperty (§7, Table7–8). 

---

# 🧪 theoryinnovationpropertyverification (1–10score)

1. whetherexistin**completeallphasesame** MCRPS/D/K system？**1/10** (this paperisparallelroundturn+NN/SAscheduling, andmultiplelayervertical—batchquantity—scoreflow—pressuretransfercombinationphasedistanceveryfar). 
2. whetherhave**verticalspacescorelayer**queueingmodeling？**0/10** (pureplaneparallel). 
3. whetherhave**inverted pyramidcapacityallocationplacement**theory？**0/10**. 
4. whetherhave**relatedtoreach+batchquantityservice+Poissonscoreflow**combination？**1/10** (proposeandrelatedtoreachstudyresearch, butthis papermodelno; alsonobatchquantity/scoreflow). 
5. whetherhave**pressuretriggermovestatetransfer**mechanism？**2/10** (haveswitchchangedecision, buttriggerlogicdifferent, andnolayerbetweenpressuremechanism). 

**verificationresults**

* ✅ **completealloriginal** (phaseforthis paper): our**"verticalscorelayer+inverted pyramidcapacity+pressuretriggerundertowardtransfer+multi-objectivereward/Ginifairness+hybridaction"**inthis paper**meannotexitappear**, thereforeandthis paperphaseratiomaintainhold**actualqualitypropertyoriginal**. 
* ⚠️ **partscorephasesimilar**: meanbelong**statedrivemoveintelligentscheduling**; this paperwith**spiritthroughnetwork + modelsimulateretreatfire**directminimizeaveragecost, andwith**MDP/strategyiteratesubstitute**doMarkovcanhusbandspecialexampleforstandard, thisonepointandour"useslearning/optimizationdoscheduling"methoddiscussionlayeraspect**cantyperatio**. 
* 🔄 **canreferencetheory**: 

 1. **MDP TabledescriptionandBellmancalculatesub/strategyiteratesubstitute** (§3, algorithm1)canisourconstruct**layerinnerortruncatebreakapproximate**theorybaseline; 
 2. **eventdrivemovesimulationframeworkunits** (§4, algorithm2)and**parameternumbersensitivefeelproperty/statisticsverify** (§6–§7, Fig4–5, Table7–8)candirectreferencetoourexperimentssection; 
 3. **NN asstrategyparameterized + SA allbureauSearch** (§5–§6)canasour DRL outer**foraccordingoptimizationdevice**. 
* ❌ **existinconflict**: nodirecttheoryconflict; **modelclosefocuspointdifferent** (this paperisplaneparallelandswitchchangecostmostsuperiorscheduling; ourstrongadjust**verticallayerlevelandcrosslayerobjectmanagemechanism**). 

---

# 💡 forourtheoryvaluevalue

1. **theoryfoundationsupport**

 * usesthis paper **MDP/strategyiteratesubstitute** provides**small-scaletruncatebreaklayerinner**baseline; inindicatenumberizationapproximateunderverificationourstrategy/rewardwhethersuperiorin MDP mostsuperior; inonegeneralscoredistributionunderuses**eventsimulation**reproduceaverageperformance. Evidence: §3 (Bellman/strategyiteratesubstitute), §4 (eventsimulation). 

2. **poordifferenceizationverification**

 * in Related Work inwiththis paperas"**parallelroundturn+NN/SA**"substituteTable, clearcertainindicateexitits**noverticallayer/nobatchquantity/noscoreflow/noinverted pyramid/nopressurecrosslayer**; ourworkworkuniquepropertybodyappearin**spacestructureandmechanismcombination**whilenonsinglepureschedulingdevicereplace. parametersee**Fig1(p.5)**, **§2–§3**. 

3. **numberlearningworktoolreference**

 * adoptingits**statecodecodeandonedimensionalmapping**idea (§3, etc.equation(9))forour**layer×queue**largestatecompress; 
 * borrowusesits**statisticsverify**flowprocess (t verify, placementinformationareabetween; §6–§7, Table7–8)comeevaluatesourstrategyin**differentscoredistribution/methodpooretc.level**understablehealthyproperty; 
 * treatits **NN parameterizedstrategy** as**foraccordingbaseline**, andour **TD7/SALE** strongizationlearningfor andcolumncompare. 

4. **citeusesstrategy**

 * **methodbaseline**: citeusesits *MDP+strategyiteratesubstitute* ascansolutionanalysisforaccording; 
 * **experimentsmethod**: citeusesits**eventsimulationalgorithm**and**statisticsverify**flowprocess; 
 * **relatedstudyresearchreview**: citeusesitsfor**polling/relatedtoreach/ML+simulation**papercontributecontext (§1). andinour"innovationpoint"placeComparisonits**nonscorelayer**setting, breakthroughexitour**verticallayerandinverted pyramidcapacity**. 

---

# ✅ mostendresultdiscussion

* **theoryinnovationdegreecertainrecognize (based onthispaper)**: **9/10**
* **ourinnovationuniqueproperty**: **completeallunique** (phaseforthis paperplacesubstituteTableplaneparallel—switchchangecost—NN/SArangeequation). 

 * attach: this paper**Fig4–5 (p.20)**and**Table7–8 (p.23–24)**showshowitsoptimizationstrategyinmultiplescoredistributionformstateunder**statisticsetc.efficiency/stablehealthyproperty**, thisisourin"scoredistributiondifferencequality/highdimensionalobservation"Experimental Designandstatisticssignificantlypropertyreportprovideed**candirectreferencerangeequation**. 

> requiresneedspeech, Icanwithtreatonaspect"citeusessentenceequationmodelversion + foraccordingTable (our↔this paper)"wholemanagebecomecandirectpasteinputdiscussionpaper Related Work andexperimentsmethodattachrecord, andprovidesoneindividual**small-scalelayerinnerMDPtruncatebreakbaseline**canreproduceexperimentsfootbook (containeventsimulationandtverifyflowprocess). 

---

**theoryinnovationrelateddegree**: **in** (schedulingoptimizationmethodlearningstrong, queueingscorelayerstructureweak)
**ourinnovationuniquepropertycertainrecognize**: **completeallunique** (phaseforthis paperrangeequation)
**suggestionadjuststudyprioritizedlevel**: **important** (asMDPbaselineandeventsimulationexperimentsmethodreference)

---

**Analysis Completion Date**: 2025-01-28 
**Analysis Quality**: Detailed analysis withMDPstrategyiteratesubstituteandspiritthroughnetworkoptimizationschedulingmechanism 
**Recommended Use**: asschedulingoptimizationmethodbaseline, referenceMDPstrategyiteratesubstituteandeventdrivemovesimulationtechnique