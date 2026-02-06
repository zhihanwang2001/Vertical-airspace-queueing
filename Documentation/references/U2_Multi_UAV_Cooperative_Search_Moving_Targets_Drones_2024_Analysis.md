# U2shouldusesscoreanalysis: multiplenohuman-machine cooperationSearchshiftmoveobjective

**Full Citation**: "Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios" (*Drones*, 2024)

---

## 📄 Application Basic Information

* **Application Domain**: **Search** (3D multiplenohuman-machine cooperationSearch"shiftmoveobjective")——proposes high-altitude/lowemptycooperativeSearcharchitecture, balancinghighemptywide coverageandlowemptyhigh recognition. seeabstractand§1. 
* **System Scale**: **small-scale (<10)**——experimentsuses **3/5/8 units UAV**, objectivenumber 10–25, grid 20×20. see§5.1, Fig.6–7. 
* **Optimization Objective**: **multi-objective**——minimize"area uncertainty"+ maximize"captureobtainobjectivenumber"; forshouldobjectivefunctionnumberandconstraintseeequation(5)–(12). 

## 🚁 UAVsystemmodelingscoreanalysis

1. **Airspace Modeling**

* **spacestructure**: **3D space** (planediscreteizationisgrid; femtorowmoveas 6 directions: N/E/S/W/on/under). see§3.1–3.2, Fig.2. 
* **Altitude Processing**: **multi-altitude (3 layer)**——discretehighdegree (z∈{1,2,3}), forshouldviewvenuelargesmallandexploretest/false alarmgeneralratesingleadjustrelationship (Table 2; equation(8)(10)(11)(12)). 
* **Conflict Avoidance**: **geometric+rules**——safealldistancedistanceconstraint (equation(9))+ **Action Mask** rulesscreenhidenoefficiency/dangeraction (§4.2, equation(14), Fig.4). 

2. **Task Scheduling Mode**

* **scoreallocationstrategy**: **CTDE hybridequation** (setinequation critic, distributed actor), modelingis **DEC-POMDP**; eachmachinebased onlocalviewvenue/allteampositiondodecision (§4.4: observeobservecontaineach UAV 3D position (O_p^t)). 
* **movestateweightscheduling**: **completeallmovestate**——eachstepsbased onobjective/uncertainty/collisionriskweightselectaction (Fig.5, Algorithm 1). 
* **load balancing**: **noshowequation** (withcovercoverandcaptureobtainismain). 

3. **systemconstraint**

* **capacitylimitation**: notmodelingloadweight/canquantity; only**sensor capability**and**viewvenuelargesmall** (Table 2). 
* **whenbetweenconstraint**: taskonlimit (T) anddiscreteplanningsteps (§5.1, 500 steps/returncombine). 
* **spaceconstraint**: edgeboundary/gridization+**mostsmalluncertaintyfusioncombine** (§3.4 equation(3)(4)), safealldistancedistance (equation(9)). 

## 🔍 andourverticalscorelayersystemComparison

**discussionpaperneedpoint**: threelayer**high–in–low**cooperative; **Rule-based underfallcaptureobtainmechanism** (inspecttesttodoubtsimilarobjectivethenfallhigh, equation(13)); **Action-mask** collisionruleavoid; **viewvenuecodecode**stableinputdimensionaldegree (Fig.3); algorithmis **AM-MAPPO** (Fig.4–5). 

**ouruniquedesign (returncustomer)**: 5 layerhighdegree {100,80,60,40,20m}; **inverted pyramidcapacity {8,6,4,3,2}**; **congestionpressuretriggercrosslayerundersink**; **29 dimensionalsystemlearningstate**; **MCRPS/D/K queuenetwork**. 

### systeminnovationpropertyComparison (1–10 score)

1. **whetherhaveverticalscorelayerUAVscheduling？**: **4/10** (have"3 layerhighdegree"andhighlowcooperative, but**nocapacity/throughchannelization**layerlevelmanagecontrol). 
2. **whetherhaveinverted pyramidresourceallocationplacement？**: **0/10** (notinvolveandlayerlevelcapacity/throughchannelnumberscoreallocation). 
3. **whetherhavequeuetheorymodelingUAVsystem？**: **1/10** (adoptinguncertainty/BayesleafsiplaceFigand MARL, **noqueueing/congestionprocess**). 
4. **whetherhavepressuretriggerlayerbetweentransfer？**: **3/10** (have**objectivetrigger**fallhighcaptureobtain, butnon**congestionpressure**trigger, also not crosslayer"scheduling/turnoperate"mechanism). 
5. **whetherhave ≥29dimensionalstatespacedesign？**: **4/10** (viewvenuecodecode 3×3 + positionetc., dimensionaldegreefollow N/FOV change, butnotreachour**29 dimensionalsystemlearningMetrics**andnonqueueizationstructure). 

### shouldusesscenariopoordifference

**existingworkworkclosefocus**: objectiveSearchcovercoverand**pathpathplanning**, shiftmoveobjectivecaptureobtain, 3D collisionruleavoid, viewvenue/uncertaintyplaceFigfusioncombine (Fig.6–9, §3.3–3.5). 
**ourinnovationpoint**: 

* ✅ **verticalairspacequeueizationmanagement** (layer/aspect/throughchannelcapacity)
* ✅ **scorelayercapacitymovestateoptimization** (inverted pyramidandcanweightallocationplacement)
* ✅ **based onqueueingtheorysystemdesign** (MCRPS/D/K)
* ✅ **highdimensionalsystemstate (29 dimensional)**and**congestionpressuretrigger**crosslayer

## 💡 forourstudyresearchvaluevalue

1. **shouldusesverificationvaluevalue**: paperinthreelayerhighlowcooperative + movestatefallhighcaptureobtain (equation(13))proofclear**highdegreelayertimes**cansignificantlyimpactcovercover/recognizedistinguishefficiency, isourdo**5 layerprecisefineizationmanagement**provideappearactualmovemachine. 
2. **methodComparisonvaluevalue**: Treatsits **AM-MAPPO** as"**noqueue/nocapacity**"strong baseline, Comparisonour**scorelayercapacity+pressuretransfer**in **p95/p99 etc.tailpartwhendelay**, overflow rateonimprovement (Fig.8–9 Metricscanrepeatuses: captureobtainnumber, covercovergridnumber, averageuncertainty, return). 
3. **scenarioextensionvaluevalue**: treatits"shiftmoveobjectiveSearch"extensionis"**scorelayerairspace + tasktoreachqueue + service/turnoperate**", studyresearch**highlayerconvergegather—lowlayerpreciseaddwork**flowmove. 
4. **performancebaselinevaluevalue**: usesits**3/5/8 units UAV**setplacementandrewardauthorityweight (Table 3)reproduceexperiments; newincrease**layerbenefitusesrate/crosslayerswitchchangetimesnumber/layercongestiondegree**etc.ourspecialhaveMetrics. 

**shouldusesinnovationdegree**: **7/10** (phasefor UAV Searchpapercontribute, thispapertreat"shiftmoveobjective + 3D + highlowcooperative + Mask collision"wholecombinegetwhen). 
**oursuperiorpotentialcertainrecognize**: **significantlyimprovement** (in"verticalscorelayercapacity+queueization+pressuretransfer"dimensionaldegreeobviousleadfirst). 

---

### implementplacesuggestion (fastforreceiveyoussystem)

* **airspace**: Treatsits 3 layerchangeis **5 layer**; treat Table 2 "viewvenue/exploretest/false alarm"mappingisour**layercapacity K_l**and"servicequalityquantity"curves, useswithbuild**inverted pyramid {8,6,4,3,2}**and**layerbetweensubstitutevalue**. 
* **mechanism**: treat"**objectivetriggerfallhigh**"replaceis"**congestionpressuretriggercrosslayer**", triggerthresholdvaluecomeselflayerinnerqueuelength/etc.waitingwhenbetween/Ginisystemnumberetc. (our 29 dimensionalinloadMetrics). 
* **algorithm**: with **AM-MAPPO** iscontrolcore (retain **Action-mask**), inactioninaddinput"**layerbetweentransfer**"discretebranch + "**layerinnerservicestrongdegree**"continuousbranch (hybridaction); rewardaddinput**whendelay/fairness/canconsume**multi-objectiveweighted. 
* **evaluatetest**: alongusesitsfourMetrics (captureobtainnumber/covercovergrid/averageuncertainty/return), newincrease**queueizationMetrics** (layerbenefitusesrate, overflow rate, crosslayertimesnumber, tailpartscorepositionwhendelay)and**resourcedisappearconsume**. 

> **FigTable/equationciteusesspeedcheck**: 
>
> * 3D scenarioandhighlowcooperative: Fig.1–2, §3.1–3.2; threelayerandparameternumberrelationship: equation(8)(10)(11)(12), Table 2 (p.16). 
> * rulesfallhighcaptureobtain: §4.1 equation(13); Action-mask collision: §4.2 equation(14), Fig.4. 
> * viewvenuecodecode: Fig.3; DEC-POMDP: §4.4; AM-MAPPO: Fig.5, Algorithm 1. 
> * experimentsandComparison: §5.1–5.3, Fig.6–9, Table 3. 

e.g.fruitrequiresneed, Icanwithtreat**"5 layer+inverted pyramidcapacity+pressuretriggertransfer" AM-MAPPO trainingcooperatediscuss**and**MetricsTable**accordingyous 29 dimensionalstatedirectprovidescanreproduceexperimentsfootbookclearsingle. 

---

**theoryinnovationrelateddegree**: **in** (have3layerairspacedesign, butlackfewcapacityizationmanagement)
**ourinnovationuniquepropertycertainrecognize**: **significantlyimprovement** (inverticalscorelayercapacityizationmethodaspect)
**suggestionadjuststudyprioritizedlevel**: **important** (asmultiplelayercooperativeSearchshouldusesreference)

---

**Analysis Completion Date**: 2025-01-28 
**Analysis Quality**: Detailed analysis withmultiplelayercooperativeSearcharchitectureandAction-maskcollisionavoidmechanism 
**Recommended Use**: asmultipleUAVcooperativeSearchshouldusesbaseline, referenceAM-MAPPOframeworkunitsandthreelayercooperativemechanism