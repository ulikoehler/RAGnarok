import { Routes } from '@angular/router';
import { ChatComponent } from './chat.component';
import { SearchComponent } from './search.component';
import { SourcesComponent } from './sources.component';
import { StatsComponent } from './stats.component';
import { ViewerComponent } from './viewer.component';
import { LoginComponent } from './login.component';
import { FiltersComponent } from './filters.component';
import { AdminComponent } from './admin.component';
import { adminGuard, authGuard } from './auth.guard';

export const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: '', redirectTo: 'chat', pathMatch: 'full' },
  { path: 'chat', component: ChatComponent, canActivate: [authGuard] },
  { path: 'chat/:threadId', component: ChatComponent, canActivate: [authGuard] },
  { path: 'search', component: SearchComponent, canActivate: [authGuard] },
  { path: 'filters', component: FiltersComponent, canActivate: [authGuard] },
  { path: 'sources', component: SourcesComponent, canActivate: [adminGuard] },
  { path: 'stats', component: StatsComponent, canActivate: [authGuard] },
  { path: 'admin', component: AdminComponent, canActivate: [adminGuard] },
  { path: 'doc/:id', component: ViewerComponent, canActivate: [authGuard] },
];
