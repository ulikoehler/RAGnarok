import { Routes } from '@angular/router';
import { ChatComponent } from './chat.component';
import { SearchComponent } from './search.component';
import { SourcesComponent } from './sources.component';
import { StatsComponent } from './stats.component';
import { ViewerComponent } from './viewer.component';

export const routes: Routes = [
  { path: '', redirectTo: 'chat', pathMatch: 'full' },
  { path: 'chat', component: ChatComponent },
  { path: 'chat/:threadId', component: ChatComponent },
  { path: 'search', component: SearchComponent },
  { path: 'sources', component: SourcesComponent },
  { path: 'stats', component: StatsComponent },
  { path: 'doc/:id', component: ViewerComponent },
];
