package diagnostics

import (
	"context"
	"net/http"
	"path/filepath"
	"sync"

	"github.com/c2h5oh/datasize"
	"github.com/ledgerwatch/erigon-lib/common"
	"github.com/ledgerwatch/erigon-lib/kv"
	"github.com/ledgerwatch/erigon-lib/kv/mdbx"
	"github.com/ledgerwatch/log/v3"
	"golang.org/x/sync/semaphore"
)

type DiagnosticClient struct {
	ctx         context.Context
	db          kv.RwDB
	metricsMux  *http.ServeMux
	dataDirPath string
	speedTest   bool

	syncStats           SyncStatistics
	snapshotFileList    SnapshoFilesList
	mu                  sync.Mutex
	headerMutex         sync.Mutex
	hardwareInfo        HardwareInfo
	peersStats          *PeerStats
	headers             Headers
	bodies              BodiesInfo
	bodiesMutex         sync.Mutex
	resourcesUsage      ResourcesUsage
	resourcesUsageMutex sync.Mutex
	networkSpeed        NetworkSpeedTestResult
	networkSpeedMutex   sync.Mutex
}

func NewDiagnosticClient(ctx context.Context, metricsMux *http.ServeMux, dataDirPath string, speedTest bool) (*DiagnosticClient, error) {
	dirPath := filepath.Join(dataDirPath, "diagnostics")
	db, err := createDb(ctx, dirPath)
	if err != nil {
		return nil, err
	}

	hInfo := ReadSysInfo(db)
	ss := ReadSyncStages(db)
	snpdwl := ReadSnapshotDownloadInfo(db)
	snpidx := ReadSnapshotIndexingInfo(db)

	return &DiagnosticClient{
		ctx:         ctx,
		db:          db,
		metricsMux:  metricsMux,
		dataDirPath: dataDirPath,
		speedTest:   speedTest,
		syncStats: SyncStatistics{
			SyncStages:       ss,
			SnapshotDownload: snpdwl,
			SnapshotIndexing: snpidx,
		},
		hardwareInfo:     hInfo,
		snapshotFileList: SnapshoFilesList{},
		bodies:           BodiesInfo{},
		resourcesUsage: ResourcesUsage{
			MemoryUsage: []MemoryStats{},
		},
		peersStats: NewPeerStats(1000), // 1000 is the limit of peers; TODO: make it configurable through a flag
	}, nil
}

func createDb(ctx context.Context, dbDir string) (db kv.RwDB, err error) {
	db, err = mdbx.NewMDBX(log.New()).
		Label(kv.DiagnosticsDB).
		WithTableCfg(func(defaultBuckets kv.TableCfg) kv.TableCfg { return kv.DiagnosticsTablesCfg }).
		GrowthStep(4 * datasize.MB).
		MapSize(16 * datasize.GB).
		PageSize(uint64(4 * datasize.KB)).
		RoTxsLimiter(semaphore.NewWeighted(9_000)).
		Path(dbDir).
		Open(ctx)
	if err != nil {
		return nil, err
	}

	return db, nil
}

func (d *DiagnosticClient) Setup() {

	rootCtx, _ := common.RootContext()

	d.setupSnapshotDiagnostics(rootCtx)
	d.setupStagesDiagnostics(rootCtx)
	d.setupSysInfoDiagnostics()
	d.setupNetworkDiagnostics(rootCtx)
	d.setupBlockExecutionDiagnostics(rootCtx)
	d.setupHeadersDiagnostics(rootCtx)
	d.setupBodiesDiagnostics(rootCtx)
	d.setupResourcesUsageDiagnostics(rootCtx)
	d.setupSpeedtestDiagnostics(rootCtx)

	//d.logDiagMsgs()
}

/*func (d *DiagnosticClient) logDiagMsgs() {
	ticker := time.NewTicker(20 * time.Second)
	quit := make(chan struct{})
	go func() {
		for {
			select {
			case <-ticker.C:
				d.logStr()
			case <-quit:
				ticker.Stop()
				return
			}
		}
	}()
}
func (d *DiagnosticClient) logStr() {
	d.mu.Lock()
	defer d.mu.Unlock()
	log.Info("SyncStatistics", "stats", interfaceToJSONString(d.syncStats))
}

func interfaceToJSONString(i interface{}) string {
	b, err := json.Marshal(i)
	if err != nil {
		return ""
	}
	return string(b)
}*/
